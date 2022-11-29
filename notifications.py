import os
import smtplib
from datetime import datetime
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

import config

def send_mail(image_path, body):
    """Sends email message of found package-detection with bbox-drawn image attached.

    Args:
        image_path: path of the drawn-on image
        body: string input of message to send
    """

    file_name = os.path.basename(os.path.normpath(image_path))
    
    msg = MIMEMultipart()
    msg['From'] = config.EMAIL_ADDRESS
    msg['To'] = config.USER_EMAIL
    msg['Subject'] = generate_subject()
    msg.attach(MIMEText(body, 'plain'))

    attachment = open(image_path, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename= %s' % file_name)
    msg.attach(part)

    server = smtplib.SMTP('smtp.office365.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()

def generate_body():
    return f"""\
Hi, this is your friendly neighborhood Package-Detector.
We've detected a package by your doorstep. 
Current Timestamp: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

We've attached an image of your detected package below.
Thank you for using our service!
"""

def generate_subject():
    return f'{date.today().strftime("%m/%d/%y")} - Package-Detector notification!'