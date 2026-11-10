# StatsBomb Events 列说明（dataset events parquet）

## 1) 说明范围
- 本文档基于文件：dataset/comp_11_season_27_Spain_La_Liga_2015_2016/events/265839.parquet
- 该文件共有 96 列。
- StatsBomb 的事件表是宽表：很多列只在某些事件类型下有值（例如传球列只在 Pass 事件有值，射门列只在 Shot 事件有值）。

## 2) 每一列含义（96 列）

| 列名 | 含义 |
|---|---|
| 50_50 | 五五球事件的结果信息（结构化字段展开后保留的摘要列）。 |
| ball_receipt_outcome | 停球/接球结果。 |
| ball_recovery_offensive | 抢回球权是否发生在进攻语境。 |
| ball_recovery_recovery_failure | 球权回收是否失败。 |
| block_deflection | 封堵后是否发生折射。 |
| block_offensive | 该封堵是否属于进攻方动作。 |
| carry_end_location | 带球结束坐标（通常是 [x, y]）。 |
| clearance_aerial_won | 解围时是否争顶成功。 |
| clearance_body_part | 解围使用的身体部位。 |
| clearance_head | 解围是否用头。 |
| clearance_left_foot | 解围是否用左脚。 |
| clearance_other | 解围是否用其他部位。 |
| clearance_right_foot | 解围是否用右脚。 |
| counterpress | 是否处于反抢情境。 |
| dribble_nutmeg | 过人是否穿裆。 |
| dribble_outcome | 过人结果（成功/失败等）。 |
| dribble_overrun | 过人是否趟大。 |
| duel_outcome | 对抗结果。 |
| duel_type | 对抗类型。 |
| duration | 该事件持续时长（秒）。 |
| foul_committed_advantage | 犯规后裁判是否给对方进攻有利。 |
| foul_committed_card | 犯规对应牌（黄/红等）。 |
| foul_committed_offensive | 是否进攻犯规。 |
| foul_committed_type | 犯规类型。 |
| foul_won_advantage | 被犯规方是否获得有利判罚。 |
| foul_won_defensive | 是否防守方赢得犯规。 |
| goalkeeper_body_part | 门将动作使用的身体部位。 |
| goalkeeper_end_location | 门将动作结束坐标。 |
| goalkeeper_outcome | 门将动作结果。 |
| goalkeeper_position | 门将动作时位置类型。 |
| goalkeeper_success_in_play | 门将处理是否成功并保持比赛进行。 |
| goalkeeper_technique | 门将技术类型。 |
| goalkeeper_type | 门将动作类型。 |
| id | 事件唯一 ID（UUID）。 |
| index | 事件在该场比赛时间序列中的索引。 |
| interception_outcome | 拦截结果。 |
| location | 事件起始坐标（通常是 [x, y]）。 |
| match_id | 比赛 ID。 |
| minute | 比赛分钟。 |
| miscontrol_aerial_won | 停球失误相关空中对抗是否赢下。 |
| off_camera | 事件是否发生在镜头外。 |
| out | 该动作后球是否出界。 |
| pass_aerial_won | 传球引发的空中对抗是否赢下。 |
| pass_angle | 传球角度。 |
| pass_assisted_shot_id | 该传球若导致射门，对应射门事件 ID。 |
| pass_body_part | 传球身体部位。 |
| pass_cross | 是否传中。 |
| pass_cut_back | 是否倒三角回传（cut-back）。 |
| pass_deflected | 传球是否被折射。 |
| pass_end_location | 传球落点坐标。 |
| pass_goal_assist | 是否助攻（直接形成进球）。 |
| pass_height | 传球高度（地面/半高/高球）。 |
| pass_inswinging | 是否内旋球。 |
| pass_length | 传球长度。 |
| pass_miscommunication | 是否出现配合误解。 |
| pass_no_touch | 传球是否无人触球。 |
| pass_outcome | 传球结果。 |
| pass_outswinging | 是否外旋球。 |
| pass_recipient | 传球接球队员名。 |
| pass_recipient_id | 传球接球队员 ID。 |
| pass_shot_assist | 是否形成射门助攻（不一定进球）。 |
| pass_switch | 是否转移球。 |
| pass_technique | 传球技术类型。 |
| pass_through_ball | 是否直塞球。 |
| pass_type | 传球类型（界外球、任意球、角球等）。 |
| period | 比赛节次（上半场/下半场/加时）。 |
| play_pattern | 进攻模式（Regular Play 等）。 |
| player | 当前事件球员名。 |
| player_id | 当前事件球员 ID。 |
| position | 当前事件球员角色标签（如 Left Back、Center Forward）。 |
| possession | 控球回合编号。 |
| possession_team | 当前控球队名。 |
| possession_team_id | 当前控球队 ID。 |
| related_events | 与该事件关联的其他事件 ID 列表。 |
| second | 比赛秒。 |
| shot_aerial_won | 射门相关空中对抗是否赢下。 |
| shot_body_part | 射门身体部位。 |
| shot_end_location | 射门终点位置（门内落点或被阻挡落点语义，按事件定义）。 |
| shot_first_time | 是否第一时间射门。 |
| shot_freeze_frame | 射门瞬间冻结帧（含周围球员位置等结构化信息）。 |
| shot_key_pass_id | 导致该射门的关键传球事件 ID。 |
| shot_one_on_one | 是否单刀情境。 |
| shot_outcome | 射门结果（进球/被扑/被封堵等）。 |
| shot_statsbomb_xg | StatsBomb 官方 xG。 |
| shot_technique | 射门技术类型。 |
| shot_type | 射门类型（Open Play、Set Piece 等）。 |
| substitution_outcome | 换人结果。 |
| substitution_outcome_id | 换人结果 ID。 |
| substitution_replacement | 换上球员名。 |
| substitution_replacement_id | 换上球员 ID。 |
| tactics | 战术信息（阵型及 lineup 等结构化字段）。 |
| team | 当前事件所属球队。 |
| team_id | 当前事件所属球队 ID。 |
| timestamp | 事件时间戳字符串（比赛内时钟，如 00:04:09.871）。 |
| type | 事件类型（Pass、Shot、Carry、Duel 等）。 |
| under_pressure | 当前事件是否在压力下完成。 |

## 3) 按连续变量与类别变量分类

说明：
- 下面按建模常见口径分类：连续数值、类别/离散、结构化复合。
- 你如果只做“连续 vs 类别”二分，建议先将“结构化复合”列展开后再二次归类。

### 3.1 连续数值列（8 列）
- duration
- pass_angle
- pass_length
- shot_statsbomb_xg
- index
- period
- minute
- second

### 3.2 类别/离散列（80 列）
- 50_50
- ball_receipt_outcome
- ball_recovery_offensive
- ball_recovery_recovery_failure
- block_deflection
- block_offensive
- clearance_aerial_won
- clearance_body_part
- clearance_head
- clearance_left_foot
- clearance_other
- clearance_right_foot
- counterpress
- dribble_nutmeg
- dribble_outcome
- dribble_overrun
- duel_outcome
- duel_type
- foul_committed_advantage
- foul_committed_card
- foul_committed_offensive
- foul_committed_type
- foul_won_advantage
- foul_won_defensive
- goalkeeper_body_part
- goalkeeper_outcome
- goalkeeper_position
- goalkeeper_success_in_play
- goalkeeper_technique
- goalkeeper_type
- id
- interception_outcome
- match_id
- miscontrol_aerial_won
- off_camera
- out
- pass_aerial_won
- pass_assisted_shot_id
- pass_body_part
- pass_cross
- pass_cut_back
- pass_deflected
- pass_goal_assist
- pass_height
- pass_inswinging
- pass_miscommunication
- pass_no_touch
- pass_outcome
- pass_outswinging
- pass_recipient
- pass_recipient_id
- pass_shot_assist
- pass_switch
- pass_technique
- pass_through_ball
- pass_type
- play_pattern
- player
- player_id
- position
- possession
- possession_team
- possession_team_id
- shot_aerial_won
- shot_body_part
- shot_first_time
- shot_key_pass_id
- shot_one_on_one
- shot_outcome
- shot_technique
- shot_type
- substitution_outcome
- substitution_outcome_id
- substitution_replacement
- substitution_replacement_id
- team
- team_id
- timestamp
- type
- under_pressure

### 3.3 结构化复合列（8 列）
- carry_end_location（坐标数组）
- goalkeeper_end_location（坐标数组）
- location（坐标数组）
- pass_end_location（坐标数组）
- related_events（事件 ID 列表）
- shot_end_location（坐标数组）
- shot_freeze_frame（冻结帧对象列表）
- tactics（阵型/lineup 结构体）

## 4) 一个真实射门行示例

以下为同一份 parquet 的一个 Shot 事件“全列数据”示例：

- source_file: dataset/comp_11_season_27_Spain_La_Liga_2015_2016/events/265839.parquet
- row_index_in_dataframe: 3711
- 50_50: null
- ball_receipt_outcome: null
- ball_recovery_offensive: null
- ball_recovery_recovery_failure: null
- block_deflection: null
- block_offensive: null
- carry_end_location: null
- clearance_aerial_won: null
- clearance_body_part: null
- clearance_head: null
- clearance_left_foot: null
- clearance_other: null
- clearance_right_foot: null
- counterpress: null
- dribble_nutmeg: null
- dribble_outcome: null
- dribble_overrun: null
- duel_outcome: null
- duel_type: null
- duration: 0.3662
- foul_committed_advantage: null
- foul_committed_card: null
- foul_committed_offensive: null
- foul_committed_type: null
- foul_won_advantage: null
- foul_won_defensive: null
- goalkeeper_body_part: null
- goalkeeper_end_location: null
- goalkeeper_outcome: null
- goalkeeper_position: null
- goalkeeper_success_in_play: null
- goalkeeper_technique: null
- goalkeeper_type: null
- id: 9ccf3726-4e60-42c1-8d98-7b781cb645be
- index: 269
- interception_outcome: null
- location: [102.7, 19.9]
- match_id: 265839
- minute: 4
- miscontrol_aerial_won: null
- off_camera: null
- out: null
- pass_aerial_won: null
- pass_angle: null
- pass_assisted_shot_id: null
- pass_body_part: null
- pass_cross: null
- pass_cut_back: null
- pass_deflected: null
- pass_end_location: null
- pass_goal_assist: null
- pass_height: null
- pass_inswinging: null
- pass_length: null
- pass_miscommunication: null
- pass_no_touch: null
- pass_outcome: null
- pass_outswinging: null
- pass_recipient: null
- pass_recipient_id: null
- pass_shot_assist: null
- pass_switch: null
- pass_technique: null
- pass_through_ball: null
- pass_type: null
- period: 1
- play_pattern: Regular Play
- player: Jordi Alba Ramos
- player_id: 5211.0
- position: Left Back
- possession: 10
- possession_team: Barcelona
- possession_team_id: 217
- related_events: [1585fac2-f169-43b3-80d0-51805b543afb, 215bbc6a-168d-403d-af4d-33d59262d377]
- second: 9
- shot_aerial_won: null
- shot_body_part: Left Foot
- shot_end_location: [110.0, 24.9]
- shot_first_time: True
- shot_freeze_frame: [{'location': array([88.4, 49.3]), 'player': {'id': 3497, 'name': 'Vicente Iborra De La Fuente'}, 'position': {'id': 19, 'name': 'Center Attacking Midfield'}, 'teammate': False}, {'location': array([103.3, 35.1]), 'player': {'id': 7154, 'name': 'Sebastián Carlos Cristóforo Pepe'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'teammate': False}, {'location': array([112.8, 30.7]), 'player': {'id': 3253, 'name': 'Adil Rami'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'teammate': False}, {'location': array([95.5, 26.4]), 'player': {'id': 11094, 'name': 'Arda Turan'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'teammate': True}, {'location': array([101.2, 45.5]), 'player': {'id': 6379, 'name': 'Sergi Roberto Carnicer'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'teammate': True}, {'location': array([112.6, 47.0]), 'player': {'id': 5246, 'name': 'Luis Alberto Suárez Díaz'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'teammate': True}, {'location': array([114.0, 35.6]), 'player': {'id': 5503, 'name': 'Lionel Andrés Messi Cuccittini'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'teammate': True}, {'location': array([108.5, 22.0]), 'player': {'id': 4320, 'name': 'Neymar da Silva Santos Junior'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'teammate': True}, {'location': array([104.0, 25.2]), 'player': {'id': 6841, 'name': 'Víctor Machín Pérez'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'teammate': False}, {'location': array([110.2, 24.9]), 'player': {'id': 6672, 'name': 'Jorge Andújar Moreno'}, 'position': {'id': 2, 'name': 'Right Back'}, 'teammate': False}, {'location': array([118.5, 36.9]), 'player': {'id': 6823, 'name': 'Sergio Rico González'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'teammate': False}, {'location': array([110.5, 44.9]), 'player': {'id': 5737, 'name': "Steven N''Kemboanza Mike Christopher Nzonzi"}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'teammate': False}, {'location': array([110.5, 40.8]), 'player': {'id': 10476, 'name': 'Timothée Kolodziejczak'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'teammate': False}, {'location': array([110.1, 33.9]), 'player': {'id': 27483, 'name': 'Benoît Trémoulinas'}, 'position': {'id': 6, 'name': 'Left Back'}, 'teammate': False}]
- shot_key_pass_id: null
- shot_one_on_one: null
- shot_outcome: Blocked
- shot_statsbomb_xg: 0.009876591
- shot_technique: Normal
- shot_type: Open Play
- substitution_outcome: null
- substitution_outcome_id: null
- substitution_replacement: null
- substitution_replacement_id: null
- tactics: null
- team: Barcelona
- team_id: 217
- timestamp: 00:04:09.871
- type: Shot
- under_pressure: null

## 5) 使用建议（建模角度）
- 直接喂模型的连续特征通常优先用：period、minute、second、duration、location、各类 end_location、pass_length、pass_angle、shot_statsbomb_xg 等。
- 文本类别列（type、position、play_pattern、pass_type、shot_type 等）需编码后再用。
- 结构化列表列（related_events、shot_freeze_frame、tactics）通常需要展开/聚合后再作为特征使用。
- ID 列（id、player_id、team_id、match_id）要谨慎处理，避免引入身份泄漏。
