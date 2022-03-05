from argparse import ArgumentParser
from global_variables import *
import boto3
from pathlib import Path
import os
import boto
import boto.s3
import sys
from boto.s3.key import Key
from pystache.tests.spectesting import yaml
import boto3
import os
import boto3



def generate_dir_if_not_exists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def file_exists(path):
    return os.path.exists(path)


'''
def get_file_from_google_drive(file_id, out_file_path_name="./downloaded_files/celebrity2000.mat"):
    url = 'https://drive.google.com/uc?id=' + file_id
    gdown.download(url, out_file_path_name, quiet=False, resume=True)
'''


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='./config_files/config.yaml', help='Config .yaml file to use for training')

    # To read the data directory from the argument given
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    # To read the data directory from the argument given
    user_path = config['user_path']

    # Fill these in - you get them when you sign up for S3
    AWS_ACCESS_KEY_ID = access
    AWS_ACCESS_KEY_SECRET = secret_access_key
    # Fill in info on data to upload
    # destination bucket name
    bucket_name = 'stance-classification'
    # source directory
    sourceDir = user_path + clean_images_path
    # destination directory name (on s3)
    destDir = 'Dimitri/'

    #max size in bytes before uploading in parts. between 1 and 5 GB recommended
    MAX_SIZE = 20 * 1000 * 1000
    #size of parts when uploading in parts
    PART_SIZE = 6 * 1000 * 1000

    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET)

    bucket = conn.create_bucket(bucket_name,
            location=boto.s3.connection.Location.DEFAULT)


    uploadFileNames = []
    for (sourceDir, dirname, filename) in os.walk(sourceDir):
        uploadFileNames.extend(filename)
        break

    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    for filename in uploadFileNames:
        sourcepath = os.path.join(sourceDir + filename)
        destpath = os.path.join(destDir, filename)
        print('Uploading %s to Amazon S3 bucket %s' % \
               (sourcepath, bucket_name))

        filesize = os.path.getsize(sourcepath)
        if filesize > MAX_SIZE:
            print("multipart upload")
            mp = bucket.initiate_multipart_upload(destpath)
            fp = open(sourcepath,'rb')
            fp_num = 0
            while (fp.tell() < filesize):
                fp_num += 1
                print("uploading part %i" %fp_num)
                mp.upload_part_from_file(fp, fp_num, cb=percent_cb, num_cb=10, size=PART_SIZE)

            mp.complete_upload()

        else:
            print("singlepart upload")
            k = boto.s3.key.Key(bucket)
            k.key = destpath
            k.set_contents_from_filename(sourcepath,
                    cb=percent_cb, num_cb=10)