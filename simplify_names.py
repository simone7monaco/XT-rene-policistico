from pathlib import Path
import re
import argparse


def get_args():
        parser = argparse.ArgumentParser(description='Exchange all the filenames in the selected directory to the default form. Default name is "experiment_{date}_{treatment}_{tube}_{zstack}_{side}"')
        parser.add_argument('dir', nargs='*', default=None, help='Selected directory')
#         parser.add_argument('--dir', type=str, default=None, help='Selected directory')

        args = parser.parse_args()

        return args

all_treats = {'ctrl', 't3', 'triac', 't4', 'tetrac', 'resv', 'dbd', 'lm609', 'uo', 'dbd+t4', 'uo+t4', 'lm609+t4', 'lm609+10ug.ml', 'lm609+2.5ug.ml'}

def get_packs(filename):
    unpack = re.split(' {1,}_?|_', filename.strip())
    
    date_idx = [i for i, item in enumerate(unpack) if re.search('[0-9]{1,2}.[0-9]{1,2}.[0-9]{2,4}', item)][0]
    unpack = unpack[date_idx:]
    date = unpack[0]
    treatment = [x.upper() for x in unpack if x.lower() in all_treats][-1]

    side = [s for s in unpack if re.match('A|B', s)]
    side = side[0] if side else 'U'

    zstack = [s.lower() for s in unpack if re.match('[24]0[xX][0-9]{1,2}', s)][0]
    alt_zstack = [s for s in unpack if re.match('\([0-9]{1,}\)', s)]
    if alt_zstack: zstack = zstack.split('x')[0] + 'x' + alt_zstack[0][1:-1]
    z1, z2 = zstack.split('x')
    zstack = f"{z1}x{int(z2):02}"

    tube = [n for n in unpack if re.fullmatch('[0-9]*', n)][0]
    
    return date, treatment, tube, zstack, side
    
    
def main():
    args = get_args()
    
    if args.dir:
        for directory in args.dir:
            folder = Path('.') / directory
            if not folder.is_dir(): continue
            print(f'>> processing "{folder}"')

            for file in folder.iterdir():
                if file.is_dir(): continue
                filename = file.stem.strip()
                extension = file.suffix
                if extension not in ['.jpg', '.jpeg', '.png', '.json'] : continue
                if filename.startswith("via_project"): continue
                if filename.startswith("sizes"): continue
#                if filename.startswith("experiment"): continue
                
                # print(filename)
                date, treatment, tube, zstack, side = get_data(filename)

            #     print(f'experiment_{date}_{treatment}_{tube}_{zstack}_{side}{extension}\n')
                new_name = f'experiment_{date}_{treatment}_{tube}_{zstack}_{side}{extension}'
#                assert not (folder / new_name).exists(), "Error: Overwriting a file!"
                file.rename(folder / new_name)
    else:
        print('Specify at least one folder to process.')

if __name__ == '__main__':
    main()
