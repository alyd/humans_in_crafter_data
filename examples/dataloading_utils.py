import pandas as pd
from pathlib import Path
import re
import os
import glob
from preprocessing import add_transition_tables, add_achievement_tables, add_participant_demographics, add_transcripts

import ipdb
import numpy as np

TASKS = ['collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron','collect_sapling','collect_stone','collect_wood','defeat_skeleton','defeat_zombie','eat_cow','eat_plant','make_iron_pickaxe','make_iron_sword','make_stone_pickaxe','make_stone_sword','make_wood_pickaxe','make_wood_sword','place_furnace','place_plant','place_stone','place_table','wake_up']

FULL_TASKS = ['achievement_collect_coal', 'achievement_collect_diamond', 'achievement_collect_drink', 'achievement_collect_iron', 'achievement_collect_sapling', 'achievement_collect_stone', 'achievement_collect_wood', 'achievement_defeat_skeleton', 'achievement_defeat_zombie', 'achievement_eat_cow', 'achievement_eat_plant', 'achievement_make_iron_pickaxe', 'achievement_make_iron_sword', 'achievement_make_stone_pickaxe', 'achievement_make_stone_sword', 'achievement_make_wood_pickaxe', 'achievement_make_wood_sword', 'achievement_place_furnace', 'achievement_place_plant', 'achievement_place_stone', 'achievement_place_table', 'achievement_wake_up'] 

TIME_FIELDS = ['image', 'action', 'reward', 'done', 'discount', 'semantic', 'player_pos', 'achievement_collect_coal', 'achievement_collect_diamond', 'achievement_collect_drink', 'achievement_collect_iron', 'achievement_collect_sapling', 'achievement_collect_stone', 'achievement_collect_wood', 'achievement_defeat_skeleton', 'achievement_defeat_zombie', 'achievement_eat_cow', 'achievement_eat_plant', 'achievement_make_iron_pickaxe', 'achievement_make_iron_sword', 'achievement_make_stone_pickaxe', 'achievement_make_stone_sword', 'achievement_make_wood_pickaxe', 'achievement_make_wood_sword', 'achievement_place_furnace', 'achievement_place_plant', 'achievement_place_stone', 'achievement_place_table', 'achievement_wake_up', 'ainventory_health', 'ainventory_food', 'ainventory_drink', 'ainventory_energy', 'ainventory_sapling', 'ainventory_wood', 'ainventory_stone', 'ainventory_coal', 'ainventory_iron', 'ainventory_diamond', 'ainventory_wood_pickaxe', 'ainventory_stone_pickaxe', 'ainventory_iron_pickaxe', 'ainventory_wood_sword', 'ainventory_stone_sword', 'ainventory_iron_sword']

def load_into_dict(dataset_paths,agent_subsample):
    """human_dataset = load_into_dict(['../expl_in_crafter/datasets/kids_scored_dataset', '../expl_in_crafter/datasets/adults_scored_dataset'])"""

    data_dict = {}
    for path in dataset_paths:
        print('loading from path: ', path)
        for subdir, dirs, _ in os.walk(Path(path)):
            for dir in dirs:
                dir_name = path.split('/')[-1].split('_')[0]+'_'+dir.split('/')[-1]
                data_dict[dir_name] = {}
                for i, np_name in enumerate(sorted(glob.glob(subdir + os.sep + dir + '/*.np[yz]'))):
                    if ('seed' not in dir_name) or (i % agent_subsample == 0 ): #subsample the agent logs
                        filename = np_name.split('/')[-1]
                        data_dict[dir_name][filename] = dict(np.load(np_name, allow_pickle=True))
    return data_dict


def run_is_clean(run_row, dirname, run_name):
    """return False if run is invalid:
        -the controller wasn't working at all
        -the participant stopped playing but the RA didn't close the game before the episode ended in Game Over
        should always return True since this was used to identify and delete invalid runs"""
    len_episode = len(run_row['action'])
    # controller probably not working:
    percent_noop = np.sum(run_row['action'] == 0)/len_episode
    if percent_noop >0.98:
        print("run ", dirname, run_name, " was more than 98% no-op")
        return False
    # participant stopped playing but game still recording:
    #check last 15 seconds of episode for no-op, assuming FPS of around 3
    if len_episode > 45:
        if np.sum(run_row['action'][-45:]) == 0:
            print("in run ", dirname, run_name, " participant stopped playing at the end.")
            return False
    return True


def load_into_df(dataset_paths,agent_subsample=1):
    """agent_subsample is the subsampling factor of the number of episodes of agent data. 5 episodes were saved every 20000 steps
    """
    print('loading dataset from files... (takes about 30 secs)')
    dataset_dict = load_into_dict(dataset_paths,agent_subsample)
    all_runs_df = pd.DataFrame()

    print('loading data into a dataframe...')
    for dirname, runs_dict in dataset_dict.items():
        for run_name, run_row in runs_dict.items():
            if ("trial" in dirname) or ("actual" in dirname):
                run_row['participant'] = '_'.join(dirname.split('_')[:-1])
                # assert run_is_clean(run_row, dirname, run_name)
            else:
                run_row['participant'] = dirname
            run_row['is_trial'] = ("trial" in dirname)
            run_row['run'] = run_name
            # remove first timestep
            for tf in TIME_FIELDS:
                run_row[tf] = run_row[tf][1:]
            run_row['ep_len'] = len(run_row['action'])
            del run_row['discount']
            del run_row['done']
            if len(all_runs_df)==0:
                all_runs_df = pd.DataFrame(data={rk:[rv] for rk,rv in run_row.items()})
            else:
                run_df = pd.DataFrame(data={rk:[rv] for rk,rv in run_row.items()})
                all_runs_df = pd.concat([all_runs_df, run_df], ignore_index=True)

    # sort df
    all_runs_df = all_runs_df.sort_values(by=['participant','run'])        
    # rename the tasks
    task_labels = [task[12:] for task in FULL_TASKS] # chop off "achievement_"
    for task, task_label in zip(FULL_TASKS, task_labels):
        all_runs_df = all_runs_df.rename(columns={task:task_label})
    # get total return
    all_runs_df['return'] = all_runs_df['reward'].apply(np.sum)
    print('computing transition tables...')
    all_runs_df = add_participant_demographics(all_runs_df)
    all_runs_df = add_transition_tables(all_runs_df)
    all_runs_df = add_achievement_tables(all_runs_df)
    all_runs_df = add_transcripts(all_runs_df)
    # for each episode create binary array indicating if each task was unlocked
    all_runs_df['success_array']=all_runs_df['achievement_table'].apply(lambda x: np.array([int(x[t]>0) for t in TASKS]))

    # Add word count for each participant
    def getwordcount(utterance_list):
        if utterance_list == None:
            return None
        else: 
            return sum([len(utt.split(' ')) for utt in utterance_list])
    all_runs_df['wordcount'] = all_runs_df['selftalk'].apply(lambda x: getwordcount(x))
    return all_runs_df
