from training.trains import train
from test_utils import test_predictions
import os
import argparse
parser=argparse.ArgumentParser(description='RCNN training and testing Module')
parser. add_argument('-cp','--csv_path',type=str,metavar='',help='path of the csv file')
parser.add_argument('-ip','--image_path',type=str,metavar='',help='path of the image_directory')
parser.add_argument('-mp','--main_directory_path',type=str,metavar='',help='path of the main directory')
parser.add_argument('-simg','--sample_image',type=int,metavar='',help='Number of total data-samples')
parser.add_argument('-e','--epochs',type=int,metavar='',help='Number of epochs')
parser.add_argument('-sm','--save_model',type=bool,metavar='',help='To save Model(True/False)')
parser.add_argument('-tm','--test_model',type=bool,metavar='',help='You want to Test your model(True/False)')
parser.add_argument('-fp','--file_path',type=str,metavar='',help='Enter image path ')
args=parser.parse_args()
csv_dir='data/df.csv'
image_dir='data/images/images'
if __name__ == "__main__":
    target2label,background_class=train(args.csv_path,
                                        args.image_path,
                                        args.main_directory_path,
                                        args.epochs,
                                        args.sample_image,
                                        args.save_model)
    if args.test_model:
        print('Enter argument:')
        test_predictions(args.file_path,target2label,background_class,args.mp,show_output=True)
       
