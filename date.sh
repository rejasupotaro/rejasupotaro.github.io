#!/bin/sh

cd ./content/posts
for f in $(find . -type f)
do
  y=$(echo $f | cut -d"/" -f 2)
  m=$(echo $f | cut -d"/" -f 3)
  d=$(echo $f | cut -d"/" -f 4)
  # t=$(echo $name | cut -d"/" -f 4- | sed s/\.markdown$/\.md/)
  # echo copy $f to $dest/$t...
  # mkdir -p $dest && cp $f $dest/$t

  line=$(sed -n '4p' $f)
  if [[ $line == date* ]] ; then
    echo "delete $f <= $line"
    sed "3d $f"
  fi
  echo "date :\"$y-$m-$d\""
done
