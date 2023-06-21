import smtplib
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP_SSL as SMTP
from email.mime.text import MIMEText

OPTIONS = {
  0: 'video',
  1: 'audio',
  2: 'video_audio'
}

sender = 'video.audio.analysis@gmail.com'
USERNAME = "video.audio.analysis@gmail.com"
PASSWORD = "reniszqangerzedp"

def sendEmail(receivers, subject, video_link=None, audio_link=None, video_name=''):

  if video_link == None and audio_link != None:
      message = "Analysis for \"{}\" has been completed. See the results on the page: {}".format(video_name, audio_link)
  elif video_link != None and audio_link == None:
      message = "Analysis for \"{}\" has been completed. See the results on the page: {}".format(video_name, video_link)
  elif video_link != None and audio_link != None:
      message = "Analysis for \"{}\" has been completed.".format(video_name) + "\n\
        Video analysis results on the page: "+video_link + "\n\
        Audio analysis results on the page: "+audio_link
  # Setup the MIME
  mail = MIMEMultipart()
  mail['From'] = USERNAME
  mail['To'] = receivers
  mail['Subject'] = subject 
 
  mail.attach(MIMEText(message))

  session = smtplib.SMTP_SSL('smtp.gmail.com', 465)
  session.login(USERNAME, PASSWORD)  
  text = mail.as_string()
  session.sendmail(USERNAME, receivers, text)
  session.quit()