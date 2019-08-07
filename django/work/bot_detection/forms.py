from django import forms
class nameForm(forms.Form):
    screen_name = forms.CharField(max_length= 50, widget=forms.TextInput(attrs={'class':'screen_name','placeholder':'@screen_name'}))
    CHOICES = (('1', 'kNN',), ('2', 'Logistic Regression',), ('3', 'Tree',), ('4', 'Random Forest',), ('5', 'Gradient Boosting',),
               ('6', 'MLP',), ('7', 'My',),)
    field = forms.ChoiceField(choices=CHOICES, widget=forms.Select(attrs={'class': 'field'}))