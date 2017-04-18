# importa smtplib per l'attuale funzione di invio
import smtplib
import csv
import numpy as np

def send_mail(name, dimension):
#def inoltro_mail():
    fromaddr = 'team.wsn2@gmail.com'
    toaddrs  = 'mattia.soldan.ms@gmail.com'
    toaddrs2 = 'riccardo.belluzzo@gmail.com'

    y = np.zeros((dimension[0],dimension[1]))
    i = 0
    msg=''+ name + '\n\n'
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in reader:
            y[i, :] = row
            i += 1

    for i in xrange(y.shape[0]):
        for ii in xrange(y.shape[1]-1):
            msg += '"' + str(y[i,ii]) + '"' + ','
        msg += '"' + str(y[i,y.shape[1]-1]) + '"\n'

    msg += '\n\n'

    # Credentials (if needed)
    username = 'team.wsn2'
    password = 'siamoforti'

    # The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username, password)
    server.sendmail(fromaddr, toaddrs, msg)
    #server.sendmail(fromaddr, toaddrs2, msg)
    server.quit()
    print 'Email sent!'
