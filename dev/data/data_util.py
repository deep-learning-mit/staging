import pandas as pd
import cv2
import youtube_dl
import json
import re
import random




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


### image saving util ###

def save_frames(yt_id_list: list, time_list: list, out_dir, res='144p'):
    assert len(yt_id_list) == len(time_list)

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


def build_images(hellaswag_csv, activity_net_csv, out_dir='hellaswag_images/'):
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
    # # create hellaswag csv files
    # for name in ['train', 'test', 'val']:
    #     create_hellaswag_csv(f'raw_data/hellaswag_{name}.jsonl', f'csv/hellaswag_{name}.csv')

    # # create tiny hellaswag test set
    # create_sampled_hellaswag_csv('csv/hellaswag_train.csv', 'csv/tiny_hellaswag_train.csv', 10)

    # # build images from tiny hellaswag
    # build_images('csv/tiny_hellaswag_train.csv', 'csv/activity_net.csv')

    pass