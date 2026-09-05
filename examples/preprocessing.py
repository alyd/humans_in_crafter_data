import numpy as np
import glob
import os
from pathlib import Path

import pickle as pkl

from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
import json
from datetime import datetime

TASKS = ['collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron','collect_sapling','collect_stone','collect_wood','defeat_skeleton','defeat_zombie','eat_cow','eat_plant','make_iron_pickaxe','make_iron_sword','make_stone_pickaxe','make_stone_sword','make_wood_pickaxe','make_wood_sword','place_furnace','place_plant','place_stone','place_table','wake_up']

STATUS = ['ainventory_health', 'ainventory_food', 'ainventory_drink', 'ainventory_energy']
INV = ['ainventory_sapling', 'ainventory_wood', 'ainventory_stone', 'ainventory_coal', 'ainventory_iron', 'ainventory_diamond', 'ainventory_wood_pickaxe', 'ainventory_stone_pickaxe', 'ainventory_iron_pickaxe', 'ainventory_wood_sword', 'ainventory_stone_sword', 'ainventory_iron_sword']
ACTION_NAMES = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep', 'place_stone', 'place_table', 'place_furnace', 'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', 'make_iron_sword']
MATERIALS = ['outside map', 'water', 'grass', 'stone', 'path','sand','tree','lava','coal',
'iron','diamond','table','furnace','Player','Cow','Zombie','Skeleton','Arrow','Plant','?', 'sleeping']

def find_facing(ep_images, above, below, left, right, actions):
    """
    Return array of shape (T,) of the semantic index the agent is facing at timestep t=0,..,T-1

    Args:
        ep_images: (T, 64, 64, 3) array of RGB images
        above: (T,) array of semantic indices of the block above the agent at timestep t=0,..,T-1
        below: (T,) array of semantic indices of the block below the agent at timestep t=0,..,T-1
        left: (T,) array of semantic indices of the block to the left of the agent at timestep t=0,..,T-1
        right: (T,) array of semantic indices of the block to the right of the agent at timestep t=0,..,T-1
        actions: (T,) array of action taken at timestep t=0,..,T-1
    
    """
    block_pixel = 63//9
    mid_player_x = 63//2- block_pixel
    mid_player_y = 63//2 

    labeled_timesteps = np.zeros((ep_images.shape[0],))
    facing_obj = np.zeros((ep_images.shape[0],))
    
    all_arrow_pixels = {
        'left' : np.concatenate([ep_images[:, mid_player_x, mid_player_y-3][:, np.newaxis, :], ep_images[:, mid_player_x-1:mid_player_x+2, mid_player_y-2]], axis=1), # T x 4 x C
        'right' : np.concatenate([ep_images[:, mid_player_x, mid_player_y+3][:, np.newaxis, :], ep_images[:, mid_player_x-1:mid_player_x+2, mid_player_y+2]], axis=1), # T x 4 x C
        'above' : np.concatenate([ep_images[:, mid_player_x-3, mid_player_y][:, np.newaxis, :], ep_images[:, mid_player_x-2, mid_player_y-1:mid_player_y+2]], axis=1), # T x 4 x C
        'below' : np.concatenate([ep_images[:, mid_player_x+3, mid_player_y][:, np.newaxis, :], ep_images[:, mid_player_x+2, mid_player_y-1:mid_player_y+2]], axis=1), # T x 4 x C
    }

    for name, sem in zip(all_arrow_pixels, [left, right, above, below]):
        # print(pix)
        pix = all_arrow_pixels[name]
        is_facing = (pix[:,:,0] >= 110) & (pix[:,:,1] >= 110) & (pix[:,:,2] <= 100)  # T x 2
        is_facing = np.all(is_facing, axis=1) # T
        labeled_timesteps[is_facing] = 1
        if np.sum(facing_obj[is_facing]) > 0:
            print('currently considering direction ', name)
            all_pixel_image = np.stack(list(all_arrow_pixels.values())).swapaxes(1,2).reshape(-1, len(ep_images), 3) # 4 x T x C
            img = Image.fromarray(all_pixel_image)
            img.save('debug_episode.png')
            import pdb; pdb.set_trace()
        facing_obj[is_facing] = sem[is_facing]

    if (labeled_timesteps == 0).any():
        #print("checking that missing timesteps are sleep actions")
        missed_idxs = np.where(labeled_timesteps == 0)[0]
        missed_idxs = np.split(missed_idxs, np.where(np.diff(missed_idxs) != 1)[0]+1)
        for missed_group in missed_idxs:
            if ACTION_NAMES[actions[missed_group[0]]] != 'sleep':
                import pdb; pdb.set_trace()
            labeled_timesteps[missed_group] = 1 
            facing_obj[missed_group] = 20 #special token

    if (labeled_timesteps == 0).any():
        print("still missing some timesteps!")
        import pdb; pdb.set_trace()
    return facing_obj


class Counter0(Counter):
    def __add__(self, other):
            if not isinstance(other, Counter):
                return NotImplemented
            result = Counter0()
            for elem, count in self.items():
                newcount = count + other[elem]
                result[elem] = newcount
            for elem, count in other.items():
                if elem not in self:
                    result[elem] = count
            return result

def add_cu_tables(df: pd.DataFrame, table_name):
    """ Takes a dataframe that already has tables and converts them into cumulative tables over the episodes of each participant. """
    for participant in set(df['participant']):
        participant_df = df[df['participant']==participant]
        participant_df.sort_values(by='run') # ensure runs are in chronological order
        running_cu_table = Counter0({})
        all_cu_tables=[]
        for ep in participant_df.itertuples():
            running_cu_table = running_cu_table + Counter0(getattr(ep,table_name))
            all_cu_tables.append(dict(running_cu_table))
        df.loc[participant_df.index, 'cu_' + table_name] = all_cu_tables
    return df

def explain_ob(ob, obs_dict):
    nonzero_obs={}
    for i, ob in enumerate(ob):
        label = obs_dict[i]
        if label == 'facing':
            nonzero_obs[label]=MATERIALS[ob]
        elif ob!=0:
            nonzero_obs[label]=ob
    return nonzero_obs

def decode_transition(tran, obs_dict):
    return explain_ob(tran[0], obs_dict), ACTION_NAMES[tran[1]], explain_ob(tran[2], obs_dict)


def add_transition_tables(dataset_df: pd.DataFrame, 
                          include_inventory: bool = True,
                          include_facing: bool = True,
                          include_movement: bool = False,
                          include_status: bool = False,
                          include_stat_gain: bool = True):
    
    """
        Add transition tables for every episode in the datafrane.
        Args:
            dataframe, what to use in the state representation
        Returns:
            dataframe with a new column of transition tables
    """
    assert include_inventory or include_facing or include_status, "Must include at least one of inventory, facing, or status"
    
    transition_tables = []
    map_coverage_tables = []
    
    for ep_data in dataset_df.itertuples():
        semantic = ep_data.semantic # T x 64 x 64
        player_pos = ep_data.player_pos + 1 # T x 2; scale player position by 1
        actions = ep_data.action  # T x 1

        # get visited positions for map coverage
        map_locs = np.zeros((64, 64))
        map_locs[player_pos - 1] = 1
        map_coverage_tables.append(map_locs)

        # Pad semantic to avoid edge cases
        semantic = np.pad(semantic, ((0,0),(1,1),(1,1)), constant_values=0) # transpose to match image orientation
        # semantic = np.rot90(semantic, k=1, axes=(1,2)) # 64 x 64 x T NOTE: semantic map is rotated relative to image!!
        # import pdb; pdb.set_trace()
        # img = Image.fromarray(ep_data.image[100]).save('db.png')
        # print(semantic[100, player_pos[100,0]-5:player_pos[100,0]+5, player_pos[100,1]-5:player_pos[100,1]+5])
        try:
            below = np.diag(semantic[:, player_pos[:,0], player_pos[:,1]+1]) # T x 1
            above = np.diag(semantic[:, player_pos[:,0], player_pos[:,1]-1]) # T x 1
            left = np.diag(semantic[:, player_pos[:,0]-1, player_pos[:,1]]) # T x 1
            right = np.diag(semantic[:, player_pos[:,0]+1, player_pos[:,1]]) # T x 1
        except:
            print("Error in getting semantic labels around agent")
            import pdb; pdb.set_trace()
            img = Image.fromarray(ep_data.image[0])
            img.show()

        # Check that we parsed the correct data
        for t in range(len(semantic)):
            if t > 0:
                assert semantic[t, player_pos[t,0], player_pos[t,1]] == 13
            assert below[t] == semantic[t, player_pos[t,0], player_pos[t,1]+1]
        
        # Get array of size (T, 1) of the semantic index the agent is facing
        facing_dir = find_facing(ep_data.image, above, below, left, right, actions)

        repr = []
        repr_dict = {}
        if include_facing:
            repr_dict[len(repr)]="facing"
            repr.append(facing_dir)
        if include_inventory:
            for k in INV:
                label=' '.join(k.split('_')[1:])
                repr_dict[len(repr)] = label
                repr.append(getattr(ep_data,k)) # T x 1
        if include_status or include_stat_gain:
            statlen=len(STATUS)
            for k in STATUS:
                label=k.split('_')[-1]
                if include_status:
                    repr_dict[len(repr)] = label
                else: #include_stat_gain:
                    repr_dict[len(repr)] = "delta " + label
                repr.append(getattr(ep_data,k)) # T x 1

        repr = np.stack(repr, axis=1).astype(int) # T x rep_dim + 4
        
        if False: #plot some transitions for debugging
            plotcount=0
            for i,row in enumerate(repr[:-1]):
                if row[0] == 0:
                    seq = np.concatenate([ep_data.image[i], ep_data.image[i+1]], axis=0)
                    img = Image.fromarray(seq)
                    img.save('debug.png')
                    print(i, ep_data.participant, ep_data.run)
                    print(row, actions[i],repr[i+1])
                    plotcount+=1
                    if plotcount>1:
                        assert False

        cur_transitions_table = {}
        for (ob1, ob2, ac) in zip(repr[:-1], repr[1:], actions[1:]):
                if ac > 4 or (include_movement and ac > 0): # no movement acs 
                    if include_stat_gain: # convert status to increase in status
                        ob1,ob2 = np.copy(ob1),np.copy(ob2)
                        ob1[-statlen:]=np.maximum(ob2[-statlen:].astype(int)-ob1[-statlen:].astype(int),0)
                        ob2[-statlen:]=np.zeros(statlen).astype(int)
                    ob1, ob2 = tuple(ob1), tuple(ob2)
                    if (ob1, ac, ob2) not in cur_transitions_table:
                        cur_transitions_table[(ob1, ac, ob2)] = 1
                    else:   
                        cur_transitions_table[(ob1, ac, ob2)] += 1 

        # per-episode transition
        transition_tables.append(cur_transitions_table)

    dataset_df['transition_table'] = transition_tables
    dataset_df['map_coverage_table'] = map_coverage_tables
    dataset_df = add_cu_tables(dataset_df, 'transition_table') # add cumulative tables
    dataset_df.attrs['obs_dict'] = repr_dict
    dataset_df.attrs['decode_transition'] =  lambda x: decode_transition(x, repr_dict)
    
    return dataset_df

def add_achievement_tables(dataset_df: pd.DataFrame):
    achievement_tables = []

    for ep_data in dataset_df.itertuples():
        cur_achievement_table = {}
        for task in TASKS:
            cur_achievement_table[task] = getattr(ep_data,task)[-1]
        achievement_tables.append(cur_achievement_table)
    dataset_df['achievement_table'] = achievement_tables
    dataset_df = add_cu_tables(dataset_df, 'achievement_table') # add cumulative tables
    return dataset_df

def add_participant_demographics(dataset_df: pd.DataFrame):
    age_summary_file = '../participant_info/overall_ages_kids_filtered.jsonl'
    gameplay_exp_kids = '../participant_info/overall_gameplay_exp_kids_filtered.jsonl'
    gameplay_exp_adults = '../participant_info/overall_gameplay_exp_adults.jsonl'
    consolidated_dict = {}
    with open(age_summary_file, 'r') as f:
        for line in f.readlines():  
            cur_line = json.loads(line)
            consolidated_dict['kids_'+cur_line['participant'].lower() + '_age'] = cur_line['age']
    with open(gameplay_exp_kids, 'r') as f:
        for line in f.readlines():  
            cur_line = json.loads(line)
            consolidated_dict['kids_'+cur_line['participant'].lower() + '_mexp'] = cur_line['prior minecraft']
            consolidated_dict['kids_'+cur_line['participant'].lower() + '_gexp'] = cur_line['prior gameplay']
    with open(gameplay_exp_adults, 'r') as f:
        for line in f.readlines():  
            cur_line = json.loads(line)
            consolidated_dict['adults_'+cur_line['participant'].lower() + '_mexp'] = cur_line['prior minecraft']
            consolidated_dict['adults_'+cur_line['participant'].lower() + '_gexp'] = cur_line['prior gameplay']

    ages = []
    mexps = []
    gexps = []
    for ep in dataset_df.itertuples():
        if ep.participant + '_age' not in consolidated_dict:
            ages.append(float('nan'))
        else:
            ages.append(consolidated_dict[ep.participant + '_age'])
        if ep.participant + '_mexp' not in consolidated_dict:
            mexps.append(float('nan'))
            gexps.append(float('nan'))
        else:
            mexps.append(consolidated_dict[ep.participant + '_mexp'])
            gexps.append(consolidated_dict[ep.participant + '_gexp'])
    dataset_df['age'] = ages  
    dataset_df['minecraft_xp'] = mexps
    dataset_df['gaming_xp'] = gexps
    return dataset_df
    

def add_transcripts(dataset_df: pd.DataFrame):
    print('adding self talk transcripts:')
    adults_transcripts_folder = '../transcripts/adults_transcripts'
    kids_transcripts_folder = '../transcripts/kids_transcripts'
    consolidated_dict = {}
    for folder, group in zip([adults_transcripts_folder,kids_transcripts_folder],['adults','kids']):
        for selftalk_file in glob.glob(folder+'/*.txt'):
            utterance_lines = []
            last_timestamp = None
            with open(selftalk_file, 'r') as f:
                for line in f.readlines():  
                    if 'STARTEPISODE_TRIAL_1' in line:
                        first_timestamp = line.split('(')[1].split(')')[0]
                    if 'ENDEPISODE_ACTUAL' in line:
                        last_timestamp = line.split('(')[1][:-1].split(')')[0]
                    if 'participant:' in line:
                        utterance_lines.append(line.split('participant: ')[-1][:-1]) # remove \n at end of every line
                consolidated_dict[group+'_'+selftalk_file.split('/')[-1].split('_')[0] + '_selftalk'] = utterance_lines
                if last_timestamp is not None:
                    #print(selftalk_file,last_timestamp,first_timestamp)
                    fmt = "%M:%S"  # Format for minute:second
                    t1 = datetime.strptime(last_timestamp, fmt)
                    t2 = datetime.strptime(first_timestamp, fmt)
                    wordrate = sum([len(utt.split(' ')) for utt in utterance_lines])/(t1-t2).total_seconds()
                    consolidated_dict[group+'_'+selftalk_file.split('/')[-1].split('_')[0] + '_wordrate'] = wordrate
    selftalk_column = []
    wordrate_column = []
    for ep in dataset_df.itertuples():
        if ep.participant + '_selftalk' not in consolidated_dict:
            selftalk_column.append(None)
        else:
            selftalk_column.append(consolidated_dict[ep.participant + '_selftalk'])
        if ep.participant + '_wordrate' not in consolidated_dict:
            wordrate_column.append(None)
        else:
            wordrate_column.append(consolidated_dict[ep.participant + '_wordrate'])
    dataset_df['selftalk'] = selftalk_column
    dataset_df['WordRate'] = wordrate_column
    # Add word count for each participant
    def getwordcount(utterance_list):
        if utterance_list == None:
            return None
        else: 
            return sum([len(utt.split(' ')) for utt in utterance_list])
    dataset_df['wordcount'] = dataset_df['selftalk'].apply(lambda x: getwordcount(x))
    return dataset_df


def add_transcripts_summaries(dataset_df: pd.DataFrame):

    question_summary_files = ['transcripts/summary_stats/overall_question.jsonl', 'adult_transcripts/summary_stats/overall_question.jsonl']
    curiosity_summary_files = ['transcripts/summary_stats/overall_curious.jsonl', 'adult_transcripts/summary_stats/overall_curious.jsonl']
    goals_summary_files = ['transcripts/summary_stats/overall_goal.jsonl', 'adult_transcripts/summary_stats/overall_goal.jsonl']
    realization_summary_files = ['transcripts/summary_stats/overall_realization.jsonl', 'adult_transcripts/summary_stats/overall_realization.jsonl']
    sentiment_summary_files = ['transcripts/summary_stats/overall_sentiment.jsonl', 'adult_transcripts/summary_stats/overall_sentiment.jsonl']
    
    consolidated_dict = {}
    for question_summary_file in question_summary_files:
        with open(question_summary_file, 'r') as f:
            for line in f.readlines():  
                cur_line = json.loads(line)
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_question'] = cur_line['num_questions']
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_totalquestion'] = cur_line['num_total']
    for goals_summary_file in goals_summary_files:
        with open(goals_summary_file, 'r') as f:
            for line in f.readlines():  
                cur_line = json.loads(line)
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_goal'] = cur_line['num_questions']    
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_totalgoal'] = cur_line['num_total']    
    for curiosity_summary_file in curiosity_summary_files:
        with open(curiosity_summary_file, 'r') as f:
            for line in f.readlines():  
                cur_line = json.loads(line)
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_curiosity'] = cur_line['num_questions']
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_totalcuriosity'] = cur_line['num_total']
    for realization_summary_file in realization_summary_files:
        with open(realization_summary_file, 'r') as f:
            for line in f.readlines():  
                cur_line = json.loads(line)
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_realization'] = cur_line['num_questions']
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_totalrealization'] = cur_line['num_total']
    for sentiment_summary_file in sentiment_summary_files:
        with open(sentiment_summary_file, 'r') as f:
            for line in f.readlines():  
                cur_line = json.loads(line)
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_sentiment'] = cur_line['num_questions']
                consolidated_dict[cur_line['participant'].split('/')[-1] + '_totalsentiment'] = cur_line['num_total']

    questions = []
    curiosities = []
    goals = []
    realizations =[]
    sentiments =[]

    totalquestions = []
    totalcuriosities = []
    totalgoals = []
    totalrealizations =[]
    totalsentiments =[]

    for ep in dataset_df.itertuples():
        ep_name = ep.run
        
        if ep_name + '_question' not in consolidated_dict:
            questions.append(float('nan'))
            totalquestions.append(float('nan'))
        else:
            questions.append(consolidated_dict[ep_name + '_question'])
            totalquestions.append(consolidated_dict[ep_name + '_totalquestion'])

        if ep_name + '_goal' not in consolidated_dict:
            goals.append(float('nan'))
            totalgoals.append(float('nan'))
        else:
            goals.append(consolidated_dict[ep_name + '_goal'])
            totalgoals.append(consolidated_dict[ep_name + '_totalgoal'])

        if ep_name + '_curiosity' not in consolidated_dict:
            curiosities.append(float('nan'))
            totalcuriosities.append(float('nan'))
        else:
            curiosities.append(consolidated_dict[ep_name + '_curiosity'])
            totalcuriosities.append(consolidated_dict[ep_name + '_totalcuriosity'])
        
        if ep_name + '_realization' not in consolidated_dict:
            realizations.append(float('nan'))
            totalrealizations.append(float('nan'))
        else:
            realizations.append(consolidated_dict[ep_name + '_realization'])
            totalrealizations.append(consolidated_dict[ep_name + '_totalrealization'])

        if ep_name + '_sentiment' not in consolidated_dict:
            sentiments.append(float('nan'))
            totalsentiments.append(float('nan'))
        else:
            sentiments.append(consolidated_dict[ep_name + '_sentiment'])
            totalsentiments.append(consolidated_dict[ep_name + '_totalsentiment'])

    dataset_df['questions'] = questions
    dataset_df['curiosities'] = curiosities
    dataset_df['goals'] = goals
    dataset_df['realizations'] = realizations   
    dataset_df['sentiments'] = sentiments
    
    dataset_df['totalquestions'] = totalquestions
    dataset_df['totalcuriosities'] = totalcuriosities
    dataset_df['totalgoals'] = totalgoals
    dataset_df['totalrealizations'] = totalrealizations 
    dataset_df['totalsentiments'] = totalsentiments  

    return dataset_df