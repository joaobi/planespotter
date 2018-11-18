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

#
# ---- Dependencies ----
FROM base AS dependencies

RUN pip3 install --upgrade pip
RUN pip3 install -r planespotter/requirements.txt

# ------------------------
# SSH Server support for ASE
# ------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && echo "root:Docker!" | chpasswd

EXPOSE 2222 80


FROM base AS release
COPY --from=dependencies . .

EXPOSE 5001

ENTRYPOINT ["/bin/bash"]
WORKDIR /planespotter
RUN chmod 755 start.sh
CMD [ "start.sh" ]

