## Intrinsically-Motivated Humans and Agents in Open-World Exploration

Data collected from adults, children and AI agents playing the simplified version of Crafter here: https://github.com/alyd/simple-crafter 

### Participants
Each child participant's identity is consistently denoted by either *m#* or *part#*, which are two completely disjoint sets of participants. The adult participants' identities are denoted by *part#* and there is no relation between *part#* adult participants and *part#* child participants. Demographic information is saved in the `participant_info` folder.

### Gameplay Logs
In the gameplay logs, for the humans the `_trial` folders contain logs from the first (approximately) two minutes of gameplay, and the `_actual` folders contain the remaining gameplay from each participant. The AI folder contains logs from four AI agents, 3 trained with DQN and one that takes random actions. The random agent's gameplay was randomly partitioned into "participant" folders with a similar distribution of amount of gameplay time as the humans. All other AI agents' logs are in folders corresponding to different random training seeds. Each seed was trained for 1,000,000 steps; 5 evaluation episodes were recorded every 20,000 training steps.

### Transcripts
The `transcripts` folder contains manually transcribed utterances from the humans during gameplay, which were then manually filtered for self-talk (private speech). Unintelligible utterances were transcribed as "#". Not all participants have transcripts (when there was no consent to record audio). Empty transcript files signify that the participant did not make any utterances that were classified as self-talk while they were playing. For the children's transcripts, start and end timestamps for each episode are also marked in the transcripts, so the utterances can be matched with the corresponding episode in the gameplay logs. 
