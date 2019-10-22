for file in long_playthrough/*
do
  mv -- "$file" "${file// /_}"
done
