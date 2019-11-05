import os
import shutil

if __name__ == '__main__':
    root = '/media/palm/Media/ptt/new3/Paper Plate'
    for maindir in os.listdir(root):
        # for clsdir in os.listdir(os.path.join(root, maindir)):
            # for subdir in os.listdir(os.path.join(root, maindir, clsdir)):
                for file in os.listdir(os.path.join(root, maindir)):
                    srcdir = os.path.join(root, maindir)
                    dstname = f'{srcdir}/{maindir + file}'
                    shutil.move(os.path.join(srcdir, file),
                                dstname
                                )
