
import csv
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_mail(names):
    fromaddr = 'team.wsn2@gmail.com'
    toaddrs  = 'mattia.soldan.ms@gmail.com'+'riccardo.belluzzo@gmail.com'+'nardi.giosue@gmail.com'

    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Risultatiti simulazioni Wireless.'
    msg['From'] = fromaddr
    body = 'File della simulazione'

    for n in names:
        f = file(n)
        attachment = MIMEText(f.read())
        attachment.add_header('Content-Disposition', 'attachment', filename=n)
        msg.attach(attachment)

    content = MIMEText(body, 'plain')
    msg.attach(content)

    # Credentials (if needed)
    username = 'team.wsn2'
    password = 'siamoforti'

    # The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username, password)
    server.sendmail(fromaddr, toaddrs,  msg.as_string())
    server.quit()

    print 'Email sent!'




