
jsdoc -t node_modules/ink-docstrap/template/ --lenient -c jsdoc.conf.json
ABSPATH=$(cd "$(dirname "$0")"; pwd)
perl -pi -e "s?$ABSPATH/??g" ./out/*.html
