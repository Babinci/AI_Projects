from django.db import models


# Create your models here.
class CypherArenaPerplexityDeepResearch(models.Model):
    data_response = models.JSONField(default=dict)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    search_type = models.CharField(null=True, blank=True, max_length=255)  ##deep_research, normal_search
    news_source = models.CharField(null=True, blank=True, max_length=255)  ##news, showbiznes, sport, tech, science, politics


class Prompt(models.Model):
    prompt = models.TextField()
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
