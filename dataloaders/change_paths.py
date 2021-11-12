import json
import argparse
import os
import os.path as osp
import tqdm

def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--image-dir', type=str, help='Image directory', default='../datasets/refuge/images/')
    parser.add_argument('--json-dir', type=str, help='Json directory', default='../datasets/refuge/labels/')
    parser.add_argument('--out-dir', type=str, help='Output directory to prevent overwrite', default='../datasets/refuge/labels_convert/')
    parser.add_argument('--sub-dir', action='store_true', default=False, help='Whether have sub-dir?')

    args = parser.parse_args()

    return args


def process_one_json(args, j):
    if args.out_dir == 'null':
        out_dir = args.json_dir
    else:
        out_dir = args.out_dir

    json_file = osp.join(args.json_dir ,j)
  
    try:
        json_list = json.load(open(json_file, 'r+'))
    except:
        print('Error in loading file at: %s'%(json_file))
        return

    try:
        ip = json_list[0]['img_path']
    except:
        print('KeyError in loading file at %s'%(json_file))
        return

    # for jpg issue
    # rel_ip = '/'.join(ip.split('/')[-1:])
    # rel_ip = rel_ip.split('.')[-2]
    # split = j.split('/')[-2]
    # new_ip = osp.join(args.image_dir, split, rel_ip+'.jpg')
    rel_ip = '/'.join(ip.split('/')[-2:])
    new_ip = osp.join(args.image_dir, rel_ip)
   
    if not osp.exists(new_ip):
        print('Image at %s could not be found!'%(new_ip))
 
    for i in range(len(json_list)):
        json_list[i]['img_path'] = new_ip

    json.dump(json_list, open(osp.join(out_dir, j), 'w'))


def check_and_create(args, dir):

    ow = args.out_dir != 'null'
    if ow and not osp.exists(dir):
        os.makedirs(dir)


def main():

    args = arguments()
    splits = ['train', 'test']
    all_jsons = []

    check_and_create(args, args.out_dir)

    for s in splits:

        assert s in os.listdir(args.json_dir), 'Could not find %s split in the dataset folder!'%(s)

        json_dir = osp.join(args.json_dir, s)  # d  labels/train
        save_dir = osp.join(args.out_dir, s)  # s_dir  processed/train
        check_and_create(args, save_dir)

        # data organizition
        if args.sub_dir is True:
            subs = os.listdir(json_dir)  # cities   labels/train
            for sub in subs:
                s_dir = osp.join(save_dir, sub)  #labels/train/A
                check_and_create(args, s_dir)
                jsons = os.listdir(osp.join(json_dir, sub))  #labels/train/A list
                jsons = [osp.join(s, sub, j) for j in jsons] #train/A/files
                all_jsons.extend(jsons)
        else:
            jsons = os.listdir(json_dir)
            jsons = [osp.join(s, j) for j in jsons]
            all_jsons.extend(jsons) # train/file

    for j in tqdm.tqdm(all_jsons, desc='Process JSON'):
        process_one_json(args, j)

if __name__ == '__main__':
    main()
