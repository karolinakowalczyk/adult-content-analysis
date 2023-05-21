import smtplib
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP_SSL as SMTP
from email.mime.text import MIMEText

sender = ''
USERNAME =''
PASSWORD = ''

def sendEmail(receivers, subject, link):

  message = "Analysis has been completed. See the results on the page: "+link
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