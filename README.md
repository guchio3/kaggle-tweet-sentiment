## kaggle-tweet-sentiment

## usage
 1. run commands
     - shell           : `docker-compose run shell`
         - ex. debug using pudb
     - python commands : `docker-compose run python {something.py}`
         - train       : `docker-compose run python train -e e001`
         - train using checkpoint : `docker-compose run python train -e e001 -c ./chekpoints/0/temp_ckpt.pth`
         - train debug : `docker-compose run python train -e e001 --debug`
         - train debug cpu : `docker-compose run python train -e e001 --debug -d cpu`
         - predict     : `not implemented`
     - notebooks       : `docker-compose run --service-ports jn`
 1. submission
     1. make tools dataset
         - ``
     1. make trained weights dataset
         - ``
     1. write kernel
     1. submit
