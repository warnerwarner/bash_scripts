import time
import os

output_dir = 'test_output.txt'
for i in range(100):
    f = open(output_dir, 'a')
    f.write(i)
    f.write(' ')
    f.write(time.time())
    f.write('\n')
    f.close()
    time.sleep(10)
