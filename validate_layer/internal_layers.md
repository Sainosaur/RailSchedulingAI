# Validation Layer (VL) Documentation
Update as code changes; draft number: 1.

## 1. Why have a VL?
The AI is a non-deterministic black box. This layers provides the determinism so the AI can undergo aggressive learning whilst ensuring no accidents. Then why have PPO?

## 2. Internal Architecture 
Why convoy (same direction traffic) model only for prototype? Also apply 4-aspect signalling.

### Layer 1: Ingestion & Static Limits
* **Inputs:** `current_v` (m/s), `current_x` (m), `proposed_a` (m/s²), `propsed_v` (m/s²), `signal_aspect`. Hazards deteted?
* **Target:** Distance to Zone (`DTZ`) — the coordinate of the next red signal or occupied block. Why train-to-zone and not train-to-train (would increase throughput). Time to Zone (TTZ) - time taken to traverse DTZ at speed limit
* **Logic:** Identify current segment and zone and override `proposed values` if the resulting velocity would exceed the segment's speed limit. How are zones decided?

### Layer 2: Simulation
* **Equation:** s_next = s_curr + (u t) + (0.5 a t²)
* **Steps:** Run three 10-second projections (T+5, T+10, T+15). Why steps? why not just use 30 ?
* **Justification:** Prevents "overshooting" into a restricted speed zone or a stop point that is currently out of the immediate sensor range.

### Layer 3: Multi-Factor Violation Check
* **Spatial Violation:** Is s_projected >= DTZ. Then ok. Jusitfy.
* **Temporal Violation:** Is TTZ + 5.0 s<= (DTZ / v). (The 5-second reaction buffer). Then ok. Jusitfy.
* **Segment Violation:** Does the train enter a new segment at a speed higher than that segment's limit?


### Layer 4: The Decision Node (4a/4b)
* **4a (Pass):** Return `proposed values` to the environment. Log: `Status: Clear`.
* **4b (Fail):** Intercept the action (quickly sieve through actions by decrementing available options and repeat layer 2 and 3 until pass found). Trigger Layer 5.

### Layer 5: XAI Log
* **XAI Logging:** Append to `override_log.csv`:
  * `timestamp`, `original_ppo_a`, `corrected_a`, `constraint_id` (e.g., "S3_Limit", "DTA_Violation").

## 3. Data Structures for Coding

### Segment Map (Lookup Table)
| Segment | Start (m) | End (m) | Limit (km/h) | Spacial Headway | Temporal Headway | 
                                                  (no buffer)       (no buffer)
                                                        
| S0      | 0000      | 02000   | 90           | 312.5           | 12.5 |

| S1      |  2000     | 04000   | 90           | 312.5           | 12.5 |

| S2      | 4000      | 06000   | 60           | 138.9           | 08.3 |

| S3      | 6000      | 10000   | 30           | 034.7           | 04.2 |

| S4      | 6000      | 10000   | 30           | 034.7           | 04.2 |

| S5      | 6000      | 10000   | 30           | 034.7           | 04.2 |

Table values rounded to nearest decimal place where possible. 
How do I justify segments, speed limits, SH and TH?

### 4. Safety Constants
* **Emergency Deceleration (a_service):** 1.0 m/s², note: Need updates from mechanical group.
* **Temporal Headway Buffer:** 5.0 s
* **Simulation Step:** 5 s intervals. Changeable, but immutable once set.
