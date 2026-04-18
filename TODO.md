# AI Scheduler — TODO Tracker


| ID | Status | Priority | Area | Description | Source |
|---|---|---|---|---|---|
| #01 | 🔴 Open | 🔴 High | Project Scope | Add timetable — transition from "following" simulator to true scheduling system with a static/configurable timetable the PPO agent learns to adhere to. | TODO_main:1 |
| #02 | ✅ Done | 🔴 High | Environment | Make better model of ideal front (lead) train — realistic physics with segment-aware speed limits and SUVAT acceleration/deceleration. | TODO_main:2 |
| #03 | 🔴 Open | 🟡 Medium | Environment | Verify code regarding punctuality — ensure scheduled/actual arrival time logic is correctly wired and producing meaningful rewards. | TODO_main:3 |
| #04 | 🔴 Open | 🟡 Medium | Reward | Justify the headway variable in the reward function — document the chosen thresholds and confirm they match operational railway standards. | TODO_main:4 |
| #05 | 🔴 Open | ⚪ Low | Docs / Tests | Update docs and unit tests to reflect all recent changes to the environment, validate layer, reward function, and server. | TODO_main:5 |
| #06 | ✅ Done | 🔴 High | Environment | Allow modelling of hazards (e.g. landslides) — expose via dashboard API and physically impact both trains through signal degradation and braking. | TODO_main:6 |
| #07 | 🔴 Open | 🔴 High | Server | Kill/restore endpoint broken — /api/dashboard/kill/kill and /restore are not reliably stopping/restarting the simulation due to environment-related issues. | TODO_main:7 |
| #08 | 🔴 Open | 🔴 High | Server | Ensure server files correctly work with the upgraded environment (ModernizedLine104 changes may have broken server ↔ env interface). | TODO_main:8 |
| #09 | 🔴 Open | ⚪ Low | Graph | Replace hardcoded station elevations in graph.py with live OpenTopoData API calls (currently hardcoded to avoid slow HTTP imports). | TODO_main:9 | graph/graph.py:26 |
| #10 | ✅ Done | 🟡 Medium | Server | Add support for WebSocket-based lead train status communication — broadcast stall/hold/speed state to dashboard in real time. | TODO_main:10 |
| #11 | 🔴 Open | 🔴 High | Validate Layer | Support multi-agent RL — validate layer currently manages only one AI train and one lead train, limiting multi-train scenarios. | TODO_main:11 |
| #12 | 🔴 Open | ⚪ Low | Validate Layer | Find colour codes used by Polish railway (ETCS/signalling) for correct signal aspect representation. | validate_layer/todo.txt |
| #13 | 🔴 Open | ⚪ Low | Validate Layer | Find distances between the ballises on track for accurate fixed-block spacing in the validator. | validate_layer/todo.txt |
| #14 | 🔴 Open | ⚪ Low | Validate Layer | Find justification for why ETCS Level 3 is not used in the modelled scenario. | validate_layer/todo.txt |
| #15 | 🔴 Open | 🟡 Medium | Validate Layer | DTZ computation: towards segment boundary, if last zone before next segment is < SH, merge last two segments together (may need hard-coded handling). | validate_layer/validator.py:113 |
| #16 | 🔴 Open | 🟡 Medium | Server | Implement /api/dashboard/recommendations endpoint — currently returns "Not Implemented". | server/main.py:202 |
