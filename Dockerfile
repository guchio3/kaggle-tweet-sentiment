FROM gcr.io/kaggle-gpu-images/python

# set env
ENV LC_ALL "en_US.UTF-8"

# set dotfiles
WORKDIR /root
RUN git clone https://github.com/guchio3/guchio_utils.git
RUN rm .bashrc && ln -s /root/guchio_utils/.bashrc .bashrc
RUN rm .config/pudb/pudb.cfg &&ln -s /root/guchio_utils/pudb/pudb.cfg .config/pudb/pudb.cfg

# install git-completion.bash and git-prompt.sh
RUN wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -O ~/.git-completion.bash \
        && chmod a+x ~/.git-completion.bash
RUN wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh -O ~/.git-prompt.sh \
        && chmod a+x ~/.git-prompt.sh

# install conda packages
RUN conda install -y -c conda-forge pudb
# too slow cuda, so re-install torch
# https://github.com/pytorch/pytorch/issues/27807
#RUN conda uninstall pytorch
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch --force-reinstall

# install pip packages
RUN pip install nlpaug torchcontrib
RUN pip install transformers==2.9.0 tokenizers==0.7.0 torch==1.5.0

# pre-install transformers models
RUN python -c "from transformers import BertModel, BertTokenizer; BertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased');"
RUN python -c "from transformers import RobertaModel, RobertaTokenizer; RobertaModel.from_pretrained('roberta-base'); RobertaTokenizer.from_pretrained('roberta-base');"
RUN python -c "from transformers import RobertaModel, RobertaTokenizer; RobertaModel.from_pretrained('roberta-large'); RobertaTokenizer.from_pretrained('roberta-large');"


# set jupyter notebook
# jupyter vim key-bind settings
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN mkdir -p $(jupyter --data-dir)/nbextensions
RUN git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding
RUN jupyter nbextension enable vim_binding/vim_binding
# edit vim_bindings setting as I can use C-c for exitting insert mode
RUN sed -i "s/      'Ctrl-C': false,  \/\/ To enable clipboard copy/\/\/      'Ctrl-C': false,  \/\/ To enable clipboard copy/g" $(jupyter --data-dir)/nbextensions/vim_binding/vim_binding.js
