#!/bin/sh

# use application directory, regardless of WORKDIR defined by container
# cd /usr/src/app
cd /planespotter/web_svc/

## add custom steps here, such as:
##   fetching secrets from the remote service and exporting them to the environment
##   downloading frequently changing files from the data provider
##   and basically anything else

# exec and the absence of ­the -d option are important
# exec rails server -­b 0.0.0.0
# exec python3 prediction_api.py &
# exec python3 prediction_web.py &

# Start the first process
python3 prediction_api.py &
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start the Prediction API: $status"
  exit $status
fi

# Start the second process
python3 prediction_web.py
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start the Prediction WebServer: $status"
  exit $status
fi