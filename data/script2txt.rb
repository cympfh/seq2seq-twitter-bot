#!/usr/bin/env ruby

$buf = []

def flush
  if $buf.size > 1
    for i in 0...$buf.size
      puts $buf[i]
      if 0 < i and i < $buf.size - 1
        puts $buf[i]
      end
    end
  end
  $buf = []
end

readlines.each{|line|
  line = line.chomp
  if line[0] != '#'
    if line == ''
      flush
    else
      $buf << line
    end
  end
}

flush
