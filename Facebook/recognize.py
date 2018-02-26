from pprint import pprint
from fbrecog import FBRecog
path = 'test.jpg' # Insert your image file path here
path2 = 'test2.jpg' # Insert your image file path here

# To get these values, follow the steps on https://github.com/samj1912/fbrecog
access_token = 'xxx' # Insert your access token obtained from Graph API explorer here
cookie = 'xxx' # Insert your cookie string here
fb_dtsg = 'xxx' # Insert the fb_dtsg parameter obtained from Form Data here.

# Instantiate the recog class
# print(path)
recog = FBRecog(access_token, cookie, fb_dtsg)
# Recog class can be used multiple times with different paths
print(recog.recognize(path))
print(recog.recognize(path2))

# Call recognize_raw to get more info about the faces detected, including their positions
# pprint(recog.recognize_raw(path), indent=2)