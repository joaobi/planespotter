FROM ubuntu:latest as base
LABEL author="Joao Bilhim"

ADD start.sh planespotter/
ADD sshd_config planespotter/
ADD apps planespotter/apps
ADD libs planespotter/libs
ADD models planespotter/models
ADD web_svc planespotter/web_svc  
ADD requirements.txt planespotter/
RUN apt-get update && apt-get install -y python3 python3-pip curl
#RUN apt-get update && apt-get install -y unzip && pip install -r requirements.txt && curl -l "https://sajoaobidata.blob.core.windows.net/planepicturesblob/trainvalEKKENHOZQFSQ.zip?sp=r&st=2018-09-11T11:30:00Z&se=2020-09-11T19:30:00Z&spr=https&sv=2017-11-09&sig=HMNmsk%2BJ%2FBXLsS5bsBz9iW6DTr3%2FJEQmgV3NQi5CVl0%3D&sr=b" -o trainvalEKKENHOZQFSQ.zip && unzip trainvalEKKENHOZQFSQ.zip


#
# ---- Dependencies ----
FROM base AS dependencies
# install node packages
#RUN npm set progress=false && npm config set depth 0
#RUN npm install --only=production 
# copy production node_modules aside
#RUN cp -R node_modules prod_node_modules
# install ALL node_modules, including 'devDependencies'
RUN pip3 install --upgrade pip
RUN pip3 install -r planespotter/requirements.txt
#RUN curl -l "https://sajoaobidata.blob.core.windows.net/planepicturesblob/trainvalEKKENHOZQFSQ.zip?sp=r&st=2018-09-11T11:30:00Z&se=2020-09-11T19:30:00Z&spr=https&sv=2017-11-09&sig=HMNmsk%2BJ%2FBXLsS5bsBz9iW6DTr3%2FJEQmgV3NQi5CVl0%3D&sr=b" -o trainvalEKKENHOZQFSQ.zip
#RUN  unzip trainvalEKKENHOZQFSQ.zip

# ------------------------
# SSH Server support
# ------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && echo "root:Docker!" | chpasswd
# COPY /planespotter/sshd_config /etc/ssh/
EXPOSE 2222 80


FROM base AS release
COPY --from=dependencies . .

EXPOSE 5001

# CMD [ "prediction_api.py" ]

#CMD [ "python3", "./train.py" ]

ENTRYPOINT ["/bin/bash"]
WORKDIR /planespotter
RUN chmod 755 start.sh
CMD [ "start.sh" ]

# ENTRYPOINT [ "python3" ]
# WORKDIR /planespotter
# CMD [ "python3 /planespotter/web_svc/prediction_api.py" ; "python3 /planespotter/web_svc/prediction_web.py" ]
