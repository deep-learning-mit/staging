import pandas as pd
import cv2
import youtube_dl
import json
import re
import random
import os
from datasets import Dataset




### hellaswag util ###

def create_hellaswag_csv(jsonl_filepath, out_file):
    '''
    Preprocess hellaswag jsonl file from `filepath`, create and save csv into `data` directory.

    Preprocessing:
    * Keep only ActivityNet prompts
    '''
    with open(jsonl_filepath, 'r') as f:
        df = pd.read_json(f, lines=True)

    # preprocessing
    df = df.loc[df['source_id'].str.contains('activitynet')]
    df.to_csv(out_file)


def create_sampled_hellaswag_csv(hellaswag_csv, out_file, n):
    '''
    randomly sample rows from hellaswag csv to create a new csv of length n. all
    rows have been checked for their corresponding video being public
    '''
    with open(hellaswag_csv, 'r') as f:
        df = pd.read_csv(f)

    seen = set()
    ydl = youtube_dl.YoutubeDL()
    ydl.add_default_info_extractors()

    while len(seen) < n:
        m = random.randint(0, df.shape[0])
        if m in seen:
            continue
        # check if corresponding video is not private
        series = df.iloc[m]
        yt_id = re.search(r'activitynet~v_(.*)', series['source_id']).group(1)
        url = f'https://www.youtube.com/watch?v={yt_id}'
        try:
            ydl.extract_info(url, download=False)
        except youtube_dl.utils.YoutubeDLError:
            continue
        seen.add(m)

    sampled_df = df.iloc[list(seen)]
    sampled_df = sampled_df.drop(labels='Unnamed: 0', axis=1)
    sampled_df.to_csv(out_file)


def jsonl2ds(hellaswag_jsonl, return_df=False, out_file=None):
    '''
    Given a hellaswag jsonl filepath, return a huggingface Dataset containing the file data
    as objects (ie not all strings). Preprocess hellaswag jsonl file from `filepath`,
    create and save json into `data/json/` directory if `out_file` nonempty.

    Preprocessing:
    * Keep only ActivityNet prompts
    '''
    with open(hellaswag_jsonl, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    df = pd.DataFrame(data)
    df = df.loc[df['source_id'].str.contains('activitynet')]
    if out_file:
        df.to_json(out_file, orient='records', lines=True)

    ds = Dataset.from_pandas(df)
    if return_df:
        return df
    return ds


def create_sampled_hellaswag_jsonl(hellaswag_jsonl, out_file, n):
    '''
    randomly sample rows from hellaswag csv to create a new csv of length n. all
    rows have been checked for their corresponding video being public
    '''
    with open(hellaswag_jsonl, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
        df = pd.DataFrame(data)

    seen = set()
    ydl = youtube_dl.YoutubeDL()
    ydl.add_default_info_extractors()

    while len(seen) < n:
        m = random.randint(0, df.shape[0])
        if m in seen:
            continue
        # check if corresponding video is not private
        series = df.iloc[m]
        yt_id = re.search(r'activitynet~v_(.*)', series['source_id']).group(1)
        url = f'https://www.youtube.com/watch?v={yt_id}'
        try:
            ydl.extract_info(url, download=False)
        except youtube_dl.utils.YoutubeDLError:
            continue
        seen.add(m)

    sampled_df = df.iloc[list(seen)]
    sampled_df.to_json(out_file, orient='records', lines=True)


### image saving util ###

def save_frames(yt_id_list: list, time_list: list, out_dir, res='144p'):
    assert len(yt_id_list) == len(time_list)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ydl = youtube_dl.YoutubeDL()
    ydl.add_default_info_extractors()

    count_saved = 0
    for _id, time in zip(yt_id_list, time_list):
        url = 'https://www.youtube.com/watch?v=' + _id
        info = ydl.extract_info(url, download=False)
        for f in info['formats']:
            fn = f['format_note']
            if res is None or fn == res:
                url = f['url']
                cap = cv2.VideoCapture(url)

                # amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frame_num = f['fps'] * time
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
                _, frame = cap.read()
                cv2.imwrite(f'{out_dir}{_id}.png', frame)
                print(f'saved {_id}')
                count_saved += 1

                break
    print(f'{len(yt_id_list)} total videos, {count_saved} saved')


def get_time(annotations):
    '''
    return the middle of the time range specified in the first annotation.
    !!! assumes first annotation is used in hellaswag.
    if there are no annotations, return middle of full video

    example annotation:
    [{'segment': [0.01, 123.42336739937599], 'label': 'Fun sliding down'}]
    '''
    # replaces all single with double quotes, fine since we just need segment
    annotations = json.loads(annotations.replace("'", '"'))
    annotation = annotations[0]
    # replaces all single with double quotes, fine since we just need segment
    segment = annotation['segment']
    start, end = segment[0], segment[1]
    return (start + end) / 2


def scrape_images(hellaswag_csv, activity_net_csv, out_dir):
    '''
    scrape a frame from corresponding video of each hellaswag_csv entry,
    saved to out_dir
    '''
    hellaswag_df = pd.read_csv(hellaswag_csv)
    activity_net_df = pd.read_csv(activity_net_csv)

    yt_id_list = []
    time_list = []
    for source_id in hellaswag_df['source_id']:
        yt_id = re.search(r'activitynet~v_(.*)', source_id).group(1)
        time = get_time(activity_net_df.loc[activity_net_df['id'] == yt_id]['annotations'].item())

        yt_id_list.append(yt_id)
        time_list.append(time)
    print('start saving frames...')
    save_frames(yt_id_list, time_list, out_dir)
    print('...end saving frames')





if __name__ == '__main__':
    # ### create hellaswag csv files
    # for name in ['train', 'test', 'val']:
    #     create_hellaswag_csv(f'raw_data/hellaswag_{name}.jsonl', f'csv/hellaswag_{name}.csv')

    # ### create tiny hellaswag test set
    # create_sampled_hellaswag_csv('csv/hellaswag_train.csv', 'csv/tiny_hellaswag_train.csv', 10)

    # ### create sampled train, test, and val sets
    # csv_size_dict = {
    #     'train': 14740,
    #     'val': 3243,
    #     'test': 3521
    # }

    # sample_factor = 0.1

    # for name in ['train', 'val', 'test']:
    #     original_csv_fp = f'csv/hellaswag_{name}.csv'
    #     sampled_csv_fp = f'csv/sampled_hellaswag_{name}.csv'

    #     print(f'creating sampled {name} csv...')
    #     create_sampled_hellaswag_csv(original_csv_fp,
    #                                  sampled_csv_fp,
    #                                  csv_size_dict[name] * sample_factor)

    #     print(f'scraping {name} images...')
    #     scrape_images(sampled_csv_fp, 'csv/activity_net.csv',
    #                  f'hellaswag_images/sampled_hellaswag_{name}/')

    ### create jsonl files containing only activitynet
    for name in ['train', 'test', 'val']:
        jsonl2ds(f'../data/raw_data/hellaswag_{name}.jsonl', out_file=f'../data/jsonl/hellaswag_{name}.jsonl')

    ### create sampled train, test, and val sets
    for name in ['train']:
        create_sampled_hellaswag_jsonl('../data/jsonl/hellaswag_test.jsonl',
                                       '../data/jsonl/tiny_hellaswag_train.jsonl', 10)


    # # ### scrape images from tiny hellaswag
    # scrape_images('csv/tiny_hellaswag_train.csv', 'csv/activity_net.csv', 'hellaswag_images/tiny_hellaswag_train/')


    pass