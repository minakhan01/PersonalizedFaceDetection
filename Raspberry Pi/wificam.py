import picamera
from time import sleep
import urllib2
import urllib
import requests

camera = picamera.PiCamera()

# query_args = { 'RPI':'Active' }

url = 'http://165.227.116.88/predict'

# data = urllib.urlencode(query_args)

# request = urllib2.Request(url, data)

while True:
	camera.capture('image.jpg')
	files = {'image': open('image.jpg', 'rb')}
	request = requests.post(url, files=files)
	response = request.content
	print response
