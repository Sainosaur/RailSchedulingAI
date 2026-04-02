# Validation Layer (VL) Documentation
Update as code changes; draft number: 4.

## 1. Why have a VL?
The AI is a non-deterministic black box. This layers provides the determinism so the AI can undergo aggressive learning whilst ensuring no accidents. 
Q1) Why use ppo?
A1) PPO is black-box and non-deterministic, hence unreliable. PPO used for optimisation, VL used for safety.

## 2. Internal Architecture 
Q2) Why convoy (same direction traffic) model only for prototype? 
A2) Simplifies prototpye to one-dimensional preoblem - focus on proving AI improves throughput while managing headways.

Also, 4 aspect signalling is applied. 
Q3) Why 4-aspect signalling? This determines action space. Assume acceleration is 0.5 m/s² and emergency deceleration at 1 m/s².
A3) Green (next three zone are clear: speed limit), double-yellow (next two zone are clear : go to 60 kmph), yellow (next zone is clear: got to 30 kmph), red (next zone is occupied: go to 0 kmph). This provdes speed and decleration guidance and increases throughput because the brakes do not have to slammed near hazard, maintaining average speed for longer. Green to red is worst case senario. This does mean space is unutilised, but safety is priority and this keeps VL simple.

### Layer 1: Ingestion & Static Limits
* **Inputs:** `current_v` (m/s), `current_x` (m), `proposed_a` (m/s²), `propsed_v` (m/s²), `signal_aspect`. 
* **Target:** Distance to Zone (`DTZ`) — the coordinate of the next red signal or occupied block. Time to Zone (TTZ) - time taken to traverse DTZ at speed limit
* **Logic:** Identify current segment and zone and override `proposed values` if the resulting velocity would exceed the segment's speed limit. How are zones decided?

Q4) Why use train-to-zone (Fixed-Block Signalling)?
A4) Traditional standard (allegedly) and accessible for ETCS Level 2. Also, balises are present in the railway, but document doesn't tell where they are, so the simulation zones will be decided by designer.

### Layer 2: Simulation
* **Steps:** Run three 5-second projections (T+5, T+10, T+15). 
* **Justification:** Prevents "overshooting" into a restricted speed zone or a stop point that is currently out of the immediate sensor range.

Q5) Why 5 s intervals and use 3 steps?
A5) If we jumped to final time (linear calculation), the train would teleport through a Red-zone or an occupied zone. The steps ensure the violation is caught in the middle of the trajectory. To expand beyond this the simulation would run on a clock in which every 5 s, it would look 15 s into the future. Also, if there an improtant change like signal-aspect or newly informed occupied zone, then validate layer would run immediately.

### Layer 3: Multi-Factor Violation Check
* **Spatial Violation:** Is s_projected >= DTZ, then ok. This ensures train does not share same zone as hazard.
* **Temporal Violation:** Is TTZ + 5.0 s<= (DTZ / v), with the 5-second reaction buffer included, then ok. This ensures train are not going to fast to react to. Also, need a special case for when v is a low number or equals zero.
* **Segment Violation:** If the train enter a new segment at a speed higher than the segment's limit, the cap speed and override.


### Layer 4: The Decision Node (4a/4b)
* **4a (Pass):** Return `proposed values` to the environment. Log: `Status: Clear`.
* **4b (Fail):** Intercept the action (quickly sieve through actions by decrementing available options in action space and repeat layer 2 and 3 until pass found). Trigger Layer 5.

### Layer 5: XAI Log
* **XAI Logging:** Append to `override_log.csv`:
  * `timestamp`, `original_ppo_a`, `corrected_a`, `constraint_id` (e.g., "S3_Limit", "DTA_Violation").

## 3. Data Structures for Coding

### Segment Map (Lookup Table)
| Segment | Start (m) | End (m) | Limit (km/h) | Spacial Headway | Temporal Headway | 
                                                  (SH, no buffer)   (TH, no buffer)
                                                        
| S0      | 00582     | 05481   | 90           | 312.5           | 12.5 |

| S1      | 05481     | 14951   | 90           | 312.5           | 12.5 |

| S2      | 14951     | 37160   | 60           | 138.9           | 08.3 |

| S3      | 37160     | 47017   | 30           | 034.7           | 04.2 |

| S4      | 47017     | 67394   | 30           | 034.7           | 04.2 |

| S5      | 67394     | 76651   | 30           | 034.7           | 04.2 |

Table values rounded to nearest decimal place where possible. 
Gradients ignored in calcualtions, for now.

Q6) How do I justify segments, speed limits, SH and TH?
A6) The segments are best estimates looking at map of Line 104 release in 2025 and Excel sheet by gov with the segment speed limits becasue the numbers/distance did not match with eachother so chose them as close as possible. The speed limits are the average speed for that segment, the gov has about 26 segments in Line 104, then it was decided to group them into to 6 segments (estimated) becasue train stations we considered only for simplicity. Then, SH and TH used SUVAT formulae.

### 4. Safety Constants
* **Emergency Deceleration (a_service):** 1.0 m/s², note: Need updates from mechanical group.
* **Temporal Headway Buffer:** 5.0 s
* **Simulation Step:** 5 s intervals. Changeable, but immutable once set.

## Future Work
When writing report, then figure out estimated start up and long term cost for the VL. Also, expand by introducing moving zone and maybe opposite direction traffic.

## AI Declaration
AI was used iterativley during research and planning, and provided the template for this markdown. All final safety design choices (logic, formulae, variables, justifications) were chosen by me.
