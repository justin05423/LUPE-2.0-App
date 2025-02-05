# data_root_dir = '/Volumes/data/lupe'
data_root_dir = '/Users/justinjames/LUPE_Corder-Lab'

###### DO NOT CHANGE ANYTHING BELOW ######
behavior_names = ['still',
                  'walking',
                  'rearing',
                  'grooming',
                  'licking hindpaw L',
                  'licking hindpaw R']

behavior_colors = ['crimson',
                   'darkcyan',
                   'goldenrod',
                   'royalblue',
                   'rebeccapurple',
                   'mediumorchid']

keypoints = ["nose", "mouth", "l_forepaw", "l_forepaw_digit", "r_forepaw", "r_forepaw_digit",
             "l_hindpaw", "l_hindpaw_digit1", "l_hindpaw_digit2", "l_hindpaw_digit3",
             "l_hindpaw_digit4", "l_hindpaw_digit5", "r_hindpaw", "r_hindpaw_digit1",
             "r_hindpaw_digit2", "r_hindpaw_digit3", "r_hindpaw_digit4", "r_hindpaw_digit5",
             "genitalia", "tail_base"]

# 30 pixels = 1 cm
pixel_cm = 0.0330828

groups = [f'Group{i}' for i in range(1, 8)]

conditions = [f'Condition{i}' for i in range(1, 4)]
groups_Test = ['Combined']
conditions_Test = ['CSO17']
groups_Test2 = ['Combined']
conditions_Test2 = ['YFP']
groups_project_ACC_NociMorpDreadd_Formalin = ['Combined']
conditions_project_ACC_NociMorpDreadd_Formalin = ['EXP_YFP', 'EXP_MORP', 'EXP_CONFON']
