FILE=Gemfile.lock
if [ -f "$FILE" ]; then
    rm $FILE
fi
docker run --rm -v "$PWD:/srv/jekyll/" -p "8090:8090" \
                    -it amirpourmand/al-folio:v0.7.0 bundler  \
                    exec jekyll serve --watch --port=8090 --host=0.0.0.0 
