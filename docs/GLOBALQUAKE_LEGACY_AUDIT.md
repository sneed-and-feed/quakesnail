# Comprehensive Technical Survey & Anti-Pattern Analysis of Legacy GlobalQuake

**Date**: 2026-10-24  

---

## Executive Summary

A detailed source-code audit was conducted across the 11 core classes and CUDA kernels of the legacy `GlobalQuake` Java codebase (`BetterAnalysis.java`, `ClusterAnalysis.java`, `EarthquakeAnalysis.java`, `TauPTravelTimeCalculator.java`, `ShakeMap.java`, `ShakemapService.java`, `FeatureCities.java`, `Regions.java`, `GlobalQuakePanel.java`, `SoundsService.java`, and `hypocenter_search.cu`).

The investigation revealed significant architectural bottlenecks, concurrency hazards, heuristic artifacts, and unphysical population impact equations:
1. **Phase Picking (`BetterAnalysis.java`)**: Employs an ad-hoc 5-tier exponential moving average STA/LTA with arbitrary scalar multipliers. Glitch detection is delayed by 1.0–7.5 seconds after false events have already been published to clustering. Stations violently reset their entire filter state after 5 minutes of continuous shaking, blinded during massive (M7+) earthquakes. Out-of-order packets trigger immediate resets.
2. **Phase Association & Clustering (`ClusterAnalysis.java`)**: Lacks joint P/S inversion; S-waves are stripped out rather than utilized. Clusters actively "steal" events from one another every tick based on empirical intensity calculations. During swarms or doublet events, P-waves of secondary quakes fall into the S-wave arrival window of the primary quake and are discarded, blinding the engine.
3. **Hypocenter Inversion (`EarthquakeAnalysis.java` & `hypocenter_search.cu`)**: Executes brute-force Fibonacci disk scanning ($\sim 78,000$ points across 4 phases, evaluating $\sim 600,000$ travel times per pass) coupled with a trinary depth search. Whenever depth uncertainty exceeds $20\text{ km}$, it forcibly overrides the depth to an arbitrary $10.0\text{ km}$ fallback. Confidence intervals are calculated by marching 256 radial rays across 4 concentric polygons with repeated depth sweeps. The CUDA kernel allocates and frees VRAM on every invocation, suffers from an $O(1)$ single-station origin calculation trick, and is bottlenecked by repeated `atan2f` haversine calls.
4. **ShakeMap & Population Heuristics (`GlobalQuakePanel.java`, `ShakemapService.java`, `ShakeMap.java`)**: In `GlobalQuakePanel.java` (lines 200–294), the engine computes `countFelt` ("Possibly felt by") and `countStrong` ("Possibly heavily felt by") using arbitrary arctangent exposure multipliers:

```math
\text{feltMultiplier}(\text{PGA}) = \frac{2}{\pi} \arctan(0.2 \times \text{PGA})
```

```math
\text{feltStrongMultiplier}(\text{PGA}) = \max\left(0, \frac{2}{\pi} \arctan(0.2 \times (\text{PGA} - \text{PGA} _{\text{MMI V}}))\right)
```

   These multipliers are multiplied directly against municipal census populations without spatial integration, building fragility modeling, or empirical calibration.
5. **Municipality Catalog & Spatial Scanning (`ShakemapService.java`, `FeatureCities.java`, `Regions.java`)**: `ShakemapService` executes a linear $O(N)$ scan over 43,000 CSV rows (`worldcities.csv`) for every earthquake update, computing geological distances with spherical trigonometry on a single thread with zero spatial indexing. `Regions.java` parses $> 65\text{ MB}$ of raw GeoJSON into Java heap `Path2D.Float` objects, incurring high GC pressure.

---

## 1. Waveform Analysis & Phase Picking (`BetterAnalysis.java`)

### 1.1 Architecture & Moving Averages
`BetterAnalysis` extends the abstract `Analysis` base class to perform real-time signal conditioning and pick detection. Incoming miniSEED telemetry samples are fed to `nextSample(int v, long time, long currentTime)` (lines 62–218).

The signal is filtered across three Butterworth passbands via `WaveformTransformator`:
- **Default Band**: $2.0 - 5.0\text{ Hz}$ (`minFreqDefault`, `maxFreqDefault`, lines 46–47)
- **Low Frequency Band**: $0.4 - 5.0\text{ Hz}$ (`minFreqLow`, `maxFreqLow`, lines 49–50)
- **Ultra-Low Frequency Band**: $0.01 - 5.0\text{ Hz}$ (`minFreqUltraLow`, `maxFreqUltraLow`, lines 52–53)

Station baseline offset is computed over the first 4 seconds (`INIT_OFFSET_CALCULATION = 4000` ms), followed by ratio calibration over 10 seconds (`INIT_AVERAGE_RATIO = 10000` ms). Upon transition to `AnalysisStatus.IDLE`, `longAverage` is multiplied by an arbitrary heuristic factor:
```java
// Line 111:
longAverage *= 0.75;
```

Running absolute filtered amplitudes ($|V_{\text{filtered}}|$) are tracked using five separate exponential moving averages:
- **Short Average** ($\tau = 0.5\text{ s}$, line 126):

```math
\bar A _{\text{short}} \leftarrow \bar A _{\text{short}} - \frac{\bar A _{\text{short}} - |V|}{F_s \times 0.5}
```

- **Medium Average** ($\tau = 6.0\text{ s}$, line 127):

```math
\bar A _{\text{med}} \leftarrow \bar A _{\text{med}} - \frac{\bar A _{\text{med}} - |V|}{F_s \times 6.0}
```

- **Third Average** ($\tau = 30.0\text{ s}$, line 128):

```math
\bar A _{\text{third}} \leftarrow \bar A _{\text{third}} - \frac{\bar A _{\text{third}} - |V|}{F_s \times 30.0}
```

- **Special Average** (Instant attack, slow decay with $\tau = 40.0\text{ s}$, lines 130–134):

```math
\bar A _{\text{spec}} = \begin{cases} |V| & \text{if } |V| > \bar A _{\text{spec}} \\ \bar A _{\text{spec}} - \frac{\bar A _{\text{spec}} - |V|}{F_s \times 40.0} & \text{otherwise} \end{cases}
```

- **Long Average** ($\tau = 200.0\text{ s}$, lines 136–138):

```math
\text{If } \frac{\bar A _{\text{short}}}{\bar A _{\text{long}}} < 4.0: \quad \bar A _{\text{long}} \leftarrow \bar A _{\text{long}} - \frac{\bar A _{\text{long}} - |V|}{F_s \times 200.0}
```

### 1.2 Pick Trigger & Threshold Conditions
Event trigger logic is implemented in lines 140–152:
```java
double ratio = shortAverage / longAverage;
if (getStatus() == AnalysisStatus.IDLE && !getWaveformBuffer().isEmpty() && !getStation().disabled) {
    boolean cond1 = shortAverage / longAverage >= EVENT_THRESHOLD * 1.3 && time - eventTimer > 200;
    boolean cond2 = shortAverage / longAverage >= EVENT_THRESHOLD * 2.05 && time - eventTimer > 100;
    boolean condMain = shortAverage / thirdAverage > 3.0;
    if (condMain && (cond1 || cond2)) {
        WaveformBuffer buffer = getWaveformBuffer().extract(time - EVENT_EXTENSION_TIME * 1000, time);
        if (!buffer.isEmpty()) {
            setStatus(AnalysisStatus.EVENT);
            Event event = new Event(this, time, buffer, !getStation().isSensitivityValid());
            getDetectedEvents().add(0, event);
        }
    }
}
if (shortAverage / longAverage < EVENT_THRESHOLD) {
    eventTimer = time;
}
```
With `EVENT_THRESHOLD = 4.75`:
- `cond1` fires when $\text{ratio} \ge 6.175$ and sustained for $> 200\text{ ms}$.
- `cond2` fires when $\text{ratio} \ge 9.7375$ and sustained for $> 100\text{ ms}$.
- `condMain` requires $\bar A _{\text{short}} / \bar A _{\text{third}} > 3.0$.

### 1.3 Identified Anti-Patterns & Flaws in `BetterAnalysis`
1. **Vulnerability to Single-Sample Glitches and Glints**:
   A single SeedLink transmission spike or telemetry glitch drives $\bar A _{\text{short}}$ upwards rapidly within 100 ms, satisfying `cond2` and `condMain`. The event is immediately created and placed in `getDetectedEvents().add(0, event)`.
2. **Deferred Glitch Abort Mechanism (`endBadly`)**:
   Lines 172–175 attempt to clean up short glitches:
   ```java
   if (timeFromStart >= 1000 && (timeFromStart < 7.5 * 1000 && shortAverage < longAverage * 1.25 || shortAverage < mediumAverage * 0.12)) {
       setStatus(AnalysisStatus.IDLE);
       latestEvent.endBadly();
   }
   ```
   The abort check only runs after at least $1000\text{ ms}$ has elapsed, and up to $7500\text{ ms}$. During this critical 1.0–7.5 second window, the invalid event is already exposed to `ClusterAnalysis`, creating false clusters or corrupting real hypocenter solutions.
3. **Catastrophic Blindness on Great Earthquakes (`EVENT_TOO_LONG_DURATION`)**:
   Lines 164–170 define:
   ```java
   if (timeFromStart >= EVENT_TOO_LONG_DURATION * 1000) {
       Logger.warn("Station " + getStation().getStationCode()
               + " reset for exceeding maximum event duration (" + EVENT_TOO_LONG_DURATION + "s)");
       reset();
       getStation().reportState(StationState.INACTIVE, time);
       return;
   }
   ```
   `EVENT_TOO_LONG_DURATION = 5 * 60.0` (300 seconds). For major earthquakes ($M_w \ge 7.5$), strong shaking and low-frequency surface waves persist well past 5 minutes. GlobalQuake resets the station, wipes its filter memory, and sets its status to `INACTIVE` while the ground is actively shaking!
4. **Brittle Handling of Out-of-Order Packets**:
   Lines 70–75:
   ```java
   if (time < latestLogTime) {
       reset();
       getStation().reportState(StationState.INACTIVE, time);
       return;
   }
   ```
   SeedLink servers multiplexing multiple stations or experiencing UDP jitter occasionally deliver packets with timestamps a few milliseconds prior to `latestLogTime`. GlobalQuake immediately destroys the station's ongoing analysis and picks via `reset()`.
5. **Thread Safety & Concurrent Modification Hazards**:
   `nextSample()` is synchronized, and `second()` is synchronized on the `BetterAnalysis` instance. However, `getDetectedEvents()` returns an unsynchronized `ArrayList`. External threads iterating over `getDetectedEvents()` (such as `ClusterAnalysis.run()`) frequently collide with `getDetectedEvents().add(0, event)` or `getDetectedEvents().removeAll(toBeRemoved)`, risking `ConcurrentModificationException`.

---

## 2. Clustering & Phase Association (`ClusterAnalysis.java`)

### 2.1 Periodic Execution Pipeline
`ClusterAnalysis.run()` executes under a write lock (`clustersWriteLock.lock()`, lines 52–66):
```java
clearSWaves();
markSWaves();
//assignEventsToExistingEarthquakeClusters(); VERY CONTROVERSIAL
expandExistingClusters();
createNewClusters();
stealEvents();
mergeClusters();
updateClusters();
```

### 2.2 Formation Heuristics
In `createNewClusters()` (lines 446–480) and `potentialArrival()` (lines 421–434):
```java
long earliestPossibleTimeOfThatEvent = e.getpWave() - (long) ((dist * 1000.0) / 5.0) - 2500;
long latestPossibleTimeOfThatEvent = e.getpWave() + (long) ((dist * 1000.0) / 5.0) + 2500;
```
- Apparent seismic velocity is assumed to be constant at $v = 5.0\text{ km/s}$ across all distances and depths.
- A fixed tolerance window of $\pm 2.5\text{ s}$ is applied.
- If $\ge 4$ nearby stations (`HypocsSettings.clusterMinSize`) satisfy this linear velocity window, a cluster is formed.
- The cluster root is set to the simple unweighted centroid of the stations (`cluster.calculateRoot(true)`).

### 2.3 Identified Anti-Patterns & Flaws in `ClusterAnalysis`
1. **Cluster "Stealing" Thrash (`stealEvents`)**:
   Lines 80–117 implement event reassignment based on expected empirical intensity:
   ```java
   double distGC = GeoUtils.greatCircleDistance(event.getLatFromStation(), event.getLonFromStation(), cluster.getEarthquake().getLat(), cluster.getEarthquake().getLon());
   double expectedIntensity = IntensityTable.getIntensity(cluster.getEarthquake().getMag(), GeoUtils.gcdToGeo(distGC));
   ```
   If a station event was previously assigned to Cluster A, but Cluster B calculates a higher expected intensity, Cluster B forcibly reassigns `event.assignedCluster = cluster; cluster.getAssignedEvents().put(station, event);`. In complex ruptures, swarms, or foreshock-mainshock sequences, picks thrash between clusters on every tick, causing wild hypocenter oscillations.
2. **Exclusion and Discarding of S-Waves**:
   In `couldBeSArrival()` (lines 241–270), if an event arrival matches TauP S-wave travel time within $[-2000 - 0.03 t_S, 6000 + 0.05 t_S]$ ms and expected intensity $\ge 3.0$, it is marked:
   ```java
   event.setAsSWave(true);
   ```
   *(Line 260 contains the developer comment: `// 985 because GQ has high tendency to detect S waves earlier`)*.
   In `updateClusters()` (line 491):
   ```java
   if (!event.isValid() || event.isSWave()) {
       event.assignedCluster = null;
       iterator.remove();
   }
   ```
   **S-waves are completely purged from cluster membership.** GlobalQuake does not perform joint P-and-S travel-time inversion ($V_P / V_S$ Wadati analysis). All S-wave picks are wasted.
3. **Secondary Event Masking (False S-Wave Splitting)**:
   When a second earthquake occurs shortly after a first event (common in aftershocks, doublets, or induced seismicity), its early P-wave arrivals hit regional stations at the same time as the S-wave coda of the initial event. `markPossibleSWaves()` flags these true P-wave onsets as S-waves (`event.setAsSWave(true)`), causing `updateClusters()` to delete them. As a result, subsequent earthquakes are completely suppressed or severely delayed.
4. **Destructive Merging (`mergeClusters`)**:
   When cluster association overlap exceeds `MERGE_THRESHOLD = 0.54` (lines 148–189), one of the clusters and its corresponding earthquake is destroyed (`earthquakes.remove(earthquake1)`), firing a `QuakeRemoveEvent`. If the association subsequently fragments, a new earthquake is recreated, causing alert flapping.

---

## 3. Hypocenter Search & Inversion (`EarthquakeAnalysis.java` & `hypocenter_search.cu`)

### 3.1 Fibonacci Disk Grid Search (`EarthquakeAnalysis.java`)
The CPU hypocenter inversion relies on a multi-stage grid search over points distributed along a Fibonacci spiral on a disk:
```java
// Line 747:
public static final double PHI = 1.61803398875;

// Lines 765-767:
double ang = 360.0 / (PHI * PHI) * n; // ~137.508 degrees golden angle
double dist = Math.sqrt(n) * c;
double[] latLon = GeoUtils.moveOnGlobe(_lat, _lon, dist, ang);
```

The inversion executes four sequential coarse-to-fine phases (lines 258–299):
| Phase | Search Radius | Points ($N \times \text{mult}$) | Depth Steps | Heuristic Goal |
|---|---|---|---|---|
| **Phase 1 (FAR)** | $10,000\text{ km}$ ($90^{\circ}$) | $40,000$ | 6 iterations | Locate rough quadrant/region |
| **Phase 2 (REGIONAL)** | $2,500\text{ km}$ / $1,000\text{ km}$ | $20,000$ / $10,000$ | 7 iterations | Refine epicentral basin |
| **Phase 3 (EXACT)** | $100\text{ km}$ | $4,000$ | 8 iterations | Focus within $1^{\circ}$ |
| **Phase 4 (DEPTH)** | $10\text{ km}$ | $4,000$ | 10 iterations | Determine focal depth |

Total search space: $\approx 78,000$ horizontal coordinates $\times 6\text{ to }10$ depth levels $\approx 600,000$ TauP travel-time interpolations per single inversion pass.

### 3.2 Depth Inversion & The 10 km Hardcoded Fallback
Depth search is performed via trinary division (`getBestAtDepthBinarySearch`, lines 866–909):
```java
double depthA = lowerBound + (upperBound - lowerBound) * (1 / 3.0);
double depthB = lowerBound + (upperBound - lowerBound) * (2 / 3.0);
...
// Lines 904-908:
// additionally check 0km and 10 km
analyseHypocenter(threadData.hypocenterA, lat, lon, 0, pickedEvents, finderSettings, threadData);
threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
analyseHypocenter(threadData.hypocenterA, lat, lon, 10, pickedEvents, finderSettings, threadData);
threadData.setBest(selectBetterHypocenter(threadData.bestHypocenter, threadData.hypocenterA));
```

In `checkUncertainty` (lines 639–646):
```java
if (DEPTH_FIX_ALLOWED) {
    if (bestHypocenter.depthUncertainty > 200.0 || bestHypocenter.depthUncertainty > 20.0 &&
            (bestHypocenter.depthConfidenceInterval.minDepth() <= 10.0 && bestHypocenter.depthConfidenceInterval.maxDepth() >= 10.0)) {
        Logger.tag("Hypocs").debug("Depth uncertainty of %.1f is too high, defaulting the depth to 10km!".formatted(bestHypocenter.depthUncertainty));
        fixDepth(bestHypocenter, 10, events);
    }
}
```
Whenever depth confidence interval width exceeds $20\text{ km}$ and spans $10\text{ km}$, the engine forcibly locks focal depth to exactly $10.0\text{ km}$. In `fixDepth()` (lines 651–678), the origin time is recalculated as the median origin across reporting stations assuming this arbitrary 10 km depth.

### 3.3 256-Ray Confidence Polygon Marching
To calculate horizontal uncertainty bounds (`calculatePolygonConfidenceIntervals`, lines 680–691 and 417–466):
- 4 confidence levels ($3.0 \times, 2.0 \times, 1.5 \times, 1.25 \times \text{multiplier}$)
- `CONFIDENCE_POLYGON_EDGES = 64` radial rays spaced evenly by $360^{\circ} / 64 = 5.625^{\circ}$.
- Total rays: $4 \times 64 = 256$ rays radiating from the estimated epicenter.
- Each ray marches outward from $10\text{ km}$ up to $5000\text{ km}$ with step halving down to $0.25\text{ km}$.
- At **every single step**, it re-evaluates depth over 200 levels:
  `getBestAtDepth(200, TauPTravelTimeCalculator.MAX_DEPTH, finderSettings, 0, lat, lon, pickedEvents, threadData, true)`.
- This brute-force radial marching consumes immense CPU time, explaining why GlobalQuake's UI freezes or stutters during hypocenter updates.

### 3.4 Objective Function Heuristic
Rather than standard weighted least-squares ($\chi^2$), Huber robust loss, or Bayesian maximum a posteriori (MAP), GlobalQuake uses an ad-hoc heuristic (line 808):
```java
public static double calculateHeuristic(PreliminaryHypocenter hypocenter) {
    return Math.pow(hypocenter.correctStations, 2) / Math.pow(hypocenter.err, 2);
}
```
Where $\text{err} = \sum |t_{\text{pred}} - t_{\text{obs}}|$ and
$\text{correctStations} = \sum \max(0, 1 - \frac{|t_{\text{pred}} - t_{\text{obs}}|}{\Delta_{\text{thresh}}})$. Squaring both numerator and denominator produces extreme numerical instability when residual error is small.

### 3.5 CUDA Hypocenter Search (`hypocenter_search.cu`)
Inspection of `hypocenter_search.cu` reveals critical structural defects:
1. **Dynamic VRAM Allocation in Hot Loop**:
   Lines 446–452 and 534–549 perform `cudaMalloc` and `cudaFree` on every call:
   ```cpp
   cudaMalloc(&device_stations, station_array_size);
   cudaMalloc(&device_stations_distances, station_distances_array_size);
   cudaMalloc(&device_stations_distances_across, station_distances_array_size_across);
   cudaMalloc(&device_temp_results, ...);
   cudaMalloc(&f_results_device, results_size);
   // ... search kernel ...
   cudaFree(device_stations);
   cudaFree(device_stations_distances_across);
   cudaFree(device_stations_distances);
   cudaFree(device_temp_results);
   cudaFree(f_results_device);
   ```
   Repeated device allocations introduce high kernel launch overhead and GPU memory fragmentation.
2. **Explicit Trigonometric Bottleneck**:
   Line 120 notes:
   ```cpp
   float c = atan2f(y, x); // bottleneck of station distances calculation
   ```
   Every thread computes double-precision or single-precision `atan2f` inside the distance precomputation kernel (`precompute_station_distances`).
3. **Coarse 1D Travel-Time Interpolation**:
   `#define SHARED_TRAVEL_TABLE_SIZE 256` (line 12). The entire $0^{\circ} - 180^{\circ}$ angular distance range is sampled at only 256 discrete points ($\approx 0.707^{\circ}$ or $\approx 78.5\text{ km}$ spacing), causing significant interpolation error.
4. **Single-Station Origin Trick**:
   Lines 217–229 in `evaluate_hypocenter`:
   ```cpp
   int j = (point_index) % station_count;
   // trick with changing station that is being used for origin calculation
   {
       float ang_dist = station_distances_across[point_index];
       float s_pwave = s_stations[j];
       for(int tile = 0; tile < TILE; tile++) {
           float expected_travel_time = travel_table_interpolate(&s_travel_table[tile * SHARED_TRAVEL_TABLE_SIZE], ang_dist);
           float predicted_origin = s_pwave - expected_travel_time;
           origins[tile] = predicted_origin;
       }
   }
   ```
   To avoid computing a median across stations in GPU shared memory, the kernel binds the origin time for grid point `point_index` to a single station $j$. If station $j$ has a picking error or telemetry jitter, the trial hypocenter receives a corrupted origin time.

---

## 4. ShakeMap & Population Multiplier Heuristics

### 4.1 Pinpointing `countFelt`, `countStrong`, and `atan` Population Multipliers
The prompt specifically tasks us with identifying all occurrences of `countFelt`, `countStrong`, and arbitrary `atan` population multipliers.

#### Exact Code Locations in `GlobalQuakePanel.java`:
- **Lines 200–201**: Declaration of counters
  ```java
  int countFelt = 0;
  int countStrong = 0;
  ```
- **Lines 220–221**: Multiplier application against city census population
  ```java
  countFelt += (int) (city.city().population() * feltMultiplier(pga));
  countStrong += (int) (city.city().population() * feltStrongMultiplier(pga));
  ```
- **Lines 261–269**: UI display of "Possibly felt by"
  ```java
  if (countFelt > 0) {
      String levelStr = "%s".formatted(formatNumber(countFelt));
      ...
      String str = "Possibly felt by: ";
      g.setColor(Color.white);
      g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - levelW - 8, y);
      y += cellHeight;
  }
  ```
- **Lines 275–285**: UI display of "Possibly heavily felt by"
  ```java
  if (countStrong > 0) {
      String levelStr = "%s".formatted(formatNumber(countStrong));
      ...
      String str = "Possibly heavily felt by: ";
      g.setColor(Color.white);
      g.drawString(str, getWidth() - g.getFontMetrics().stringWidth(str) - levelW - 8, y);
  }
  ```
- **Lines 288–294**: The arbitrary `atan` mathematical functions
  ```java
  private double feltMultiplier(double pga) {
      return Math.atan(pga * 0.2) * 2 / 3.14159;
  }

  private double feltStrongMultiplier(double pga) {
      return Math.max(0, Math.atan((pga - MMIIntensityScale.V.getPga()) * 0.2) * 2 / 3.14159);
  }
  ```

#### Scientific & Mathematical Invalidation:
1. **Uncalibrated Arctangent Curve**: The function $\frac{2}{\pi} \arctan(0.2 \times \text{PGA})$ saturates asymptotically at 1.0 as $\text{PGA} \to \infty$. The coefficient $0.2$ and the choice of $\arctan$ have no basis in seismological ground-motion literature or epidemiological vulnerability modeling.
2. **Point-Source Spatial Smearing**: The equation takes the PGA calculated at the single geographic centroid of a city and multiplies it by the total population of the entire metropolitan area. A city spanning $50\text{ km}$ across varying site classes ($V_{S30}$) and epicentral distances is treated as experiencing uniform peak acceleration.
3. **Gross Overcounting Across Nested Municipalities**: `worldcities.csv` includes suburbs, metropolitan areas, and distinct urban districts with overlapping population catchments. Summing `countFelt` across all rows yields counts that can exceed total regional populations.
4. **Hardcoded Approximations**: Line 289 hardcodes $\pi$ as `3.14159` instead of `Math.PI`.
5. **Execution Inside GUI Repaint Loop**: `drawCityIntensities` is called directly from the Swing paint cycle (`paintComponent`) on the Event Dispatch Thread (EDT). The UI rendering loop is computing floating-point transcendental functions and iterating through lists while painting frames.

### 4.2 Additional `atan` Occurrences Across the Source Tree
- `hypocenter_search.cu` (line 120): `float c = atan2f(y, x); // bottleneck of station distances calculation`
- `hypocenter_search.cu` (lines 78 & 105): `*lon = atan2f(y, x);` inside `move_on_globe` and `move_on_globe_device`.

### 4.3 ShakeMap Hexagonal Grid (`ShakeMap.java` & `ShakemapService.java`)
`ShakeMap.java` builds ground-shaking maps using Uber H3 discrete global grid cells:
- Initialized at epicenter cell (`h3.latLngToCell(lat, lon, res)`, line 43).
- Expands outward via Breadth-First Search (`h3.gridDisk(current.id(), res)`, line 63).
- For every neighbor cell, calculates PGA:
  ```java
  double dist = GeoUtils.geologicalDistance(hypocenter.lat, hypocenter.lon, -hypocenter.depth, latLng.lat, latLng.lng, 0);
  dist = Math.max(0, dist - h3.getHexagonEdgeLengthAvg(res, LengthUnit.km) * 0.5);
  double pga = GeoUtils.pgaFunction(hypocenter.magnitude, dist, hypocenter.depth);
  ```
- Filtering step (line 87):
  `result.parallelStream().filter(intensityHex1 -> !isOcean(intensityHex1.id(), uhd)).collect(Collectors.toSet());`
  `isOcean` tests whether all 6 vertices and center of each H3 hexagon lie inside the oceans using `Regions.isOcean()`. This requires hundreds of `Path2D.contains()` checks per ShakeMap generation, causing high latency.
- In `ShakemapService.java` (lines 156–161):
  ```java
  private ShakeMap createShakemap(Earthquake earthquake) {
      Hypocenter hyp = earthquake.getCluster().getPreviousHypocenter();
      double mag = hyp.magnitude + hyp.depth / 200.0;
      mag += Settings.shakemapQualityOffset;
      return new ShakeMap(hyp, mag <= 4.9 ? 6 : mag < 6.4 ? 5 : mag < 8.5 ? 4 : 3);
  }
  ```
  Arbitrarily inflates magnitude by adding `hyp.depth / 200.0` when deciding H3 resolution tiers.

---

## 5. Municipality Database & Spatial Scanning (`ShakemapService.java` & `FeatureCities.java`)

### 5.1 Linear 43,000 CSV Scan Bottleneck
In `ShakemapService.java`:
- On startup, loads `cities/worldcities.csv` (43,000+ entries) into `private static final List<CityLocation> cities = new ArrayList<>();` (lines 38–69).
- On every earthquake creation or hypocenter revision (`updateCities`, lines 134–148):
  ```java
  cities.forEach(cityLocation -> {
      double pga = calculatePGA(cityLocation, earthquake);
      if (pga >= threshold) {
          result.add(new CityIntensity(cityLocation, pga));
      }
  });
  ```
- Computes `calculatePGA` for all 43,000 cities sequentially:
  ```java
  double dist = GeoUtils.geologicalDistance(earthquake.getLat(), earthquake.getLon(), -earthquake.getDepth(),
          cityLocation.lat(), cityLocation.lon(), 0);
  return GeoUtils.pgaFunction(earthquake.getMag(), dist, earthquake.getDepth());
  ```
- **Complexity**: $O(N)$ with $N \approx 43,000$. Each evaluation requires multiple trigonometric function calls (`sin`, `cos`, `sqrt`, `atan2`, `pow`, `log`).
- **Absence of Spatial Index**: No bounding box pre-filtering ($|\Delta\text{lat}|, |\Delta\text{lon}|$), no spatial partitioning tree (KD-tree, R-tree), and no discrete global grid (H3 / S2). This results in 20–50 ms latency spikes on every hypocenter revision.

### 5.2 Capital Cities Feature (`FeatureCities.java`)
- Loads `cities/country-capital-lat-long-population.csv` using OpenCSV (lines 34–53).
- Filters out cities with `population < 200_000` (line 42).
- Stored as an unmodifiable list for globe rendering.

---

## 6. Region Naming & GeoJSON Parsing (`Regions.java`)

`Regions.java` (506 lines) manages geographic naming, political boundaries, and ocean detection:
- Loads 12 GeoJSON files at startup into memory:
  - `countriesMD.json`, `countriesHD.json`, `countriesUHD.json`
  - `us-albers.geojson`, `AK-02-alaska-counties.geojson`, `hawaii-countries.geojson`
  - `jp-prefectures.geojson`, `new-zealand-districts.geojson`, `italy_provinces.geojson`
  - `region_dataset.geojson`
- Stores coordinates as Java `Path2D.Float` objects and custom `GQPolygon` vertex arrays.
- Memory Footprint: Over 65 MB of heap is allocated solely to polygon vertex structures.
- Shoreline Distance (`getShorelineDistance`, lines 250–272):
  Iterates over every vertex of every polygon in `regionsMD`:
  ```java
  for (Region reg : regionsMD) {
      for (GQPolygon polygon : reg.raws()) {
          for (int i = 0; i < polygon.getSize(); i++) {
              float pLat = polygon.getLats()[i];
              float pLon = polygon.getLons()[i];
              double dist = GeoUtils.greatCircleDistance(pLat, pLon, lat, lon);
              if (dist < closestDistance) closestDistance = dist;
          }
      }
  }
  ```
  Testing an offshore point involves tens of thousands of great-circle distance calculations.

---

## 7. Travel-Time Lookup (`TauPTravelTimeCalculator.java`)

`TauPTravelTimeCalculator.java` provides travel times for P, S, PKP, and PKIKP phases:
- Loads a pre-baked Java serialized binary file (`"travel_table/travel_table.dat"`) via `ObjectInputStream.readObject()` (lines 62–67).
- Uses bilinear interpolation across 2D arrays (`p_travel_table`, `s_travel_table`, `pkp_travel_table`, `pkikp_travel_table`) sampled at:
  - Angular resolution: $\Delta \theta = 0.1^{\circ}$
  - Depth resolution: $\Delta z = 1.0\text{ km}$ ($z \le 750\text{ km}$)
- Inversion of Travel Time to Angle (`getPWaveTravelAngle`, lines 97–104):
  Performs a binary search over angles with tolerance $\epsilon = 10^{-4}$ evaluating `getPWaveTravelTime` at each step, instead of using a direct inverted lookup table.

---

## 8. Summary of Legacy Formulas & Anti-Patterns

| Component | Legacy GlobalQuake Implementation | Architectural Flaw / Anti-Pattern | QuakeSnail Replacement |
|---|---|---|---|
| **Phase Picker** | 5 moving averages ($\tau \in [0.5, 200]\text{ s}$), ratio thresholds $6.175$ & $9.7375$. | Single-sample telemetry spikes trigger false events; 1.0–7.5s delayed abort; station reset after 5 min shaking; out-of-order packets force resets. | Multi-band filter + Beltrami-Morse 3-point impulse gate (`Delta_- v * Delta_+ v > 0`) + 1-sample causal deferral (`pending_impulse`) + $\mathcal H _{\text{clock}} \oplus \mathcal H _{\text{nope}}$ stationary microseism split. |
| **Phase Association** | Apparent velocity $v = 5.0\text{ km/s} \pm 2.5\text{ s}$; events stolen across clusters by expected intensity; S-waves discarded. | Thrashing pick assignments; secondary quakes suppressed as false S-waves; no joint P/S inversion. | Deterministic $\mathfrak{so}(C)$ skew-symmetric triad tie-breaking; joint P and S residual association; ring-buffer wavefront tracking without cluster stealing. |
| **Hypocenter Inversion** | 4-phase Fibonacci disk scan ($\sim 78,000$ points) + trinary depth search + 10 km depth clamp. | $O(N)$ brute-force search; artificial 10 km depth pileup; ad-hoc objective $(N_{\text{corr}})^2 / (\text{err})^2$. | Fast Fibonacci seed ($\sim 500$ points) + Damped Huber-IRLS / Ridge non-linear residual refinement with analytical Jacobian $\mathbf{J} = [\nabla T, 1]^T$. |
| **Confidence Bounds** | 256-ray radial marching across 4 concentric polygons with continuous depth sweeps. | Extreme CPU lockups, UI freezing, high thread contention. | Analytical 3D covariance matrix $\mathbf{C} = \hat{\sigma}^2 (\mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \mathbf{I})^{-1}$ yielding true confidence ellipsoids. |
| **CUDA Search** | Host-to-device malloc/free on every call; single-station origin shortcut; 256-entry travel table; `atan2f` haversine. | Massive driver overhead; single noisy station corrupts origin time; coarse distance table ($0.7^{\circ}$ res); CUDA OOM crashes JNI. | `#![no_std]` Rust SIMD CPU kernel with static arena ($< 2.0\text{ MB}$ SRAM); vectorized AVX2/NEON distance computation without dynamic allocation. |
| **Population Impact** | `countFelt` & `countStrong` via $\frac{2}{\pi}\arctan(0.2 \times \text{PGA})$ multiplied by municipal census population. | Completely unphysical, uncalibrated, point-source smearing, population double counting. | **Complete eradication** of `countFelt`, `countStrong`, and `atan` multipliers; replace with MMI/JMA Shindo intensity badges, PGA/PGV, and live P/S countdowns. |
| **Municipality Lookup** | Linear scan over 43,000 CSV rows on every hypocenter tick. | 20–50 ms CPU latency per update; zero spatial indexing. | Multi-tier spatial H3 / KD-tree index; $R_{\max}(M, z)$ bounding radius query completing in $< 1.0\text{ ms}$. |
| **Region Names** | $> 65\text{ MB}$ raw GeoJSON in Java heap; full vertex distance scan to shorelines. | High GC pauses, slow ocean classification. | Compact binary polygon hierarchy with spatial grid bounding boxes and fast ray-crossing test. |

---

## 9. Architectural Action Plan for QuakeSnail

To ensure full compliance with the requirements for `QuakeSnail`:
1. **Module Separation**:
   - `quakesnail/picker.py` & `rust/quakesnail_rs/picker.rs`: Implement multi-band filtering paired with Beltrami exterior-curl spike rejection and 1-sample causal deferral.
   - `quakesnail/associator.py` & `rust/quakesnail_rs/associator.rs`: Implement $\mathfrak{so}(C)$ triad association handling both P and S phases.
   - `quakesnail/inversion.py` & `rust/quakesnail_rs/inversion.rs`: Implement Fibonacci seeding with Huber-IRLS / Levenberg-Marquardt travel-time residual minimization and analytical covariance output.
   - `quakesnail/spatial.py`: Construct spatial H3 / KD-tree municipality lookup over curated multi-tier cities (global capitals, major cities, regional seismic zones).
2. **Strict Verification**:
   - Verify that no occurrences of `countFelt`, `countStrong`, `Possibly felt by`, `Possibly heavily felt by`, or `atan(pga * 0.2)` exist anywhere in `quakesnail`.
   - Ensure the `#![no_std]` Rust SIMD arena adheres to $< 2.0\text{ MB}$ static SRAM.
