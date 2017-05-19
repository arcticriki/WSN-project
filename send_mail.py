import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_mail(names):
    fromaddr = 'team.wsn2@gmail.com'
    toaddrs2  = 'mattia.soldan.ms@gmail.com'
    toaddrs = 'riccardo.belluzzo@gmail.com'
    toaddrs3 = 'nardi.giosue@gmail.com'


    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'Risultati simulazioni Wireless.'
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
    password = 'siamoforti2'

    # The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username, password)
    server.sendmail(fromaddr, toaddrs,  msg.as_string())
    server.sendmail(fromaddr, toaddrs2,  msg.as_string())
    server.sendmail(fromaddr, toaddrs3,  msg.as_string())
    server.quit()

    print 'Email sent!'



