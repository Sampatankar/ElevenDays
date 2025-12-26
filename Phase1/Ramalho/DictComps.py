"""
Two dictcomp examples from the same list of tuples
"""

DIAL_CODES = [
    (86, 'China'),
    (91, 'India'),
    (1, 'United States'),
    (62, 'Indonesia'),
    (55, 'Brazil'),
    (92, 'Pakistan'),
    (880, 'Bangladesh'),
    (234, 'Nigeria'),
    (7, 'Russia'),
    (81, 'Japan')
]

#DictComp 1:
country_code = {country: code for code, country in DIAL_CODES}
print(country_code)

#DictComp 2:
print({code: country.upper() for country, code in country_code.items() if code < 66})

