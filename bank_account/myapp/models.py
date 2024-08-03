from django.db import models

# Create your models here.

class userdetails(models.Model):
    id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=30, null=False)
    last_name = models.CharField(max_length=30)
    emailid = models.CharField(max_length=30, null=False)
    password = models.CharField(max_length=30, null=False)
    phonenumber = models.CharField(max_length=30, null=False)

    def __str__(self):
        return "%s %s %s %s %s %s" % (
            self.id, self.first_name, self.last_name, self.emailid, self.password, self.phonenumber)