from collections import Counter
import numpy as np

LVLS_TASKS={0:['collect_wood','eat_cow','collect_sapling','collect_drink','defeat_zombie','defeat_skeleton','wake_up'], 
    1:['place_table','place_plant'],
    2:['make_wood_pickaxe','make_wood_sword','eat_plant'],
    3:['collect_stone','collect_coal'],
    4:['make_stone_pickaxe','place_furnace'],
    5:['collect_iron']}

TASK_LVLS = {task: level for level, tasks in LVLS_TASKS.items() for task in tasks}

MAX_DEPTH = 6
MAX_BREADTH = 17
MAX_TASKS = 17

def get_unique_task_score(achievement_table):
    """The unique task score is the number of unique tasks a participant achieved at least once throughout all their gameplay up to this episode. 
    The achievement_table maps each task to the number of times it has been achieved"""
    return np.count_nonzero(list(achievement_table.values()))/MAX_TASKS

def get_breadth_score(achievement_table):
    """The breadth score is a measure of how broadly the player explored task space. 
    It calculates how much progress the participant made in a breadth-first traversal of the skill tree. Thus, we only count tasks up to (and including) the first incomplete level of the tree.
    The achievement_table maps each task to the number of times it has been achieved."""

    breadth_score = 0
    incomplete_lvl = False
    for lvl in range(len(LVLS_TASKS)):
        for task in LVLS_TASKS[lvl]:
            if achievement_table[task] > 0:
                breadth_score += 1
            else:
                incomplete_lvl = True
        if incomplete_lvl:
            break
    return breadth_score/MAX_BREADTH

def get_depth_score(achievement_table):
    """The depth score is a measure of how deeply the player explored task space.
    Thus function returns the depth of the deepest task achieved.
    The achievement_table maps each task to the number of times it has been achieved."""
    max_depth=0
    for task,achievement in achievement_table.items():
        if achievement > 0:
            max_depth = max(max_depth, TASK_LVLS[task]+1)
    return max_depth/MAX_DEPTH


example = """
achievement_depths = []
for ep in dataset_df.itertuples():
    achievement_depths.append(get_depth_score(ep.cu_achievement_table))
dataset_df['cu_depth_score'] = achievement_depths
#get the last depth score for each participant, which summarizes their entire playthrough
plt.hist(dataset_df.groupby(['participant'])['cu_depth_score'].last())
"""

def get_geometric_score(success_arrays):
    """Take a column or group of success arrays, calculate the success rates and return the geometric mean of the success rates
    Example to get the geometric score of each participant in the kids dataframe:
    plt.hist(kids_df.groupby(['participant'])['success_array'].apply(lambda x: get_geometric_score(x)))"""
    success_rates = success_arrays.mean()
    return np.exp(np.nanmean(np.log(1 + success_rates), -1)) - 1

def get_map_coverage(map_coverage_table):
    """Return fraction of unique map states that the participant visited.
    """
    map_coverage_table = map_coverage_table.flatten()
    return np.count_nonzero(map_coverage_table) / len(map_coverage_table)