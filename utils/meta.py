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

groups = [f'Group{i}' for i in range(1, 100)]

conditions = [f'Condition{i}' for i in range(1, 100)]
groups_demo_FormalinResponse = ['Combined']
conditions_demo_FormalinResponse = ['1Per Formalin', '5Per Formalin', 'Control']
groups_demo_LUPEAMPS_FormalinMorphine = ['Female', 'Male']
conditions_demo_LUPEAMPS_FormalinMorphine = ['Group1_0mgkg', 'Group2_0.5mgkg', 'Group3_1.0mgkg', 'Group4_5mgkg', 'Group5_10mgkg']
groups_DFNZ_DrugScreening_082025 = ['saline', 'morphine', 'fnz', 'dfnz', 'fdnz']
conditions_DFNZ_DrugScreening_082025 = ['1_dose_low', '2_dose_medium', '3_dose_high']
groups_DFNZ_SNI_082025 = ['T1_Control', 'T2_Treatment']
conditions_DFNZ_SNI_082025 = ['Saline', 'DFNZ']
groups_5xFAD_Capsaicin_2mo = ['LUPE_ ADppp1r3b_LHPcapsaicin_2mo']
conditions_5xFAD_Capsaicin_2mo = ['negative_female', 'negative_male', 'positive_female', 'positive_male']
groups_5xFAD_Capsaicin_2mo_PosNeg = ['LUPE_ ADppp1r3b_LHPcapsaicin_2mo_PosNeg']
conditions_5xFAD_Capsaicin_2mo_PosNeg = ['negative', 'positive']
groups_5xFAD_Capsaicin_3mo_MaleFemale = ['LUPE_ ADppp1r3b_LHPcapsaicin_3mo']
conditions_5xFAD_Capsaicin_3mo_MaleFemale = ['negative_female', 'negative_male', 'positive_female', 'positive_male']
groups_5xFAD_Capsaicin_4mo_MaleFemale = ['LUPE_ ADppp1r3b_LHPcapsaicin_4mo']
conditions_5xFAD_Capsaicin_4mo_MaleFemale = ['negative_female', 'negative_male', 'positive_female', 'positive_male']
groups_5xFAD_Capsaicin = ['2Month', '3Month', '4Month']
conditions_5xFAD_Capsaicin = ['negative', 'positive']
groups_5xFAD_Capsaicin_MaleFemale = ['2Month', '3Month', '4Month']
conditions_5xFAD_Capsaicin_MaleFemale = ['negative_female', 'negative_male', 'positive_female', 'positive_male']
groups_Analysis_PsiloDoseResponse_8wk = ['Combined']
conditions_Analysis_PsiloDoseResponse_8wk = ['Male', 'Female']
