#!/usr/bin/env ruby

ID=ARGV[0]

def normalize(str)
  re_url = /http[a-zA-Z0-9:_\/\.\,\-]*/
  re_at = /@[a-zA-Z0-9_]*/
  str = str.gsub(re_url, '')
  str = str.gsub(re_at, '')
  str = str.gsub(/\n/, ' ')
  str = str.strip
  if str[0] == '"' and str[-1] == '"'
    str = str.gsub(/\\"/, '"')
    str = str[1..-2]
  end
  str = str.strip
  str
end


for line in `cat #{ID}`.split "\n"
  _, id, com = line.split "\t"
  if id != "null"
    dst = normalize(`cat inreplyto/#{id}`)
    if dst != "null"
      puts dst
      puts normalize(com)
    end
  end
end
