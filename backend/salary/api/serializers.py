from rest_framework import serializers

class ResumeDataSerialazer(serializers.Serializer):
    role = serializers.CharField(max_length=150, required=True, trim_whitespace=True)
    experience_year = serializers.IntegerField(min_value=0, max_value=100,required=True)
    skills = serializers.ListField(child=serializers.CharField(max_length=100), required=False, default=[])
    region = serializers.CharField(max_length=100, required=True)
    education = serializers.CharField(max_length=200, required=False, default="Нет")

class CalculateSalarySerialazer(serializers.Serializer):
    min_salary = serializers.IntegerField()
    max_salary = serializers.IntegerField()
    recommendations = serializers.ListField(child=serializers.CharField())
