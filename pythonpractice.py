#!/usr/bin/env python3
'''
CPSC 415 -- Homework #0 support file
Stephen Davies, University of Mary Washington, fall 2019
'''

import sys
import importlib

''' 
To run this program, make sure you have a Python file in the current directory
whose name is exactly your lower-case UMW Net ID (e.g., "jsmith19") followed
by the string "_pythonpractice.py". In that file, you should have a class
defined whose name is exactly your first-letter-capitalized UMW Net ID
followed by the string "PythonPractice", as well as all the other variables
mentioned in the assignment description.

At any point (when you're done or earlier), you can run it via:

$ python3 pythonpractice.py jsmith19
'''

def print_usage():
    print('Usage: pythonpractice.py UMWNetID.')

if len(sys.argv) != 2:
    print_usage()
    sys.exit(1)

try:
    stud_module = importlib.import_module(sys.argv[1] + '_pythonpractice')
except Exception as err:
    print(str(err))
    sys.exit(2)

print("Testing {}...".format(sys.argv[1] + '_pythonpractice.py'))

points = 0

the_vars = {
    'WALL': [{'E'}],
    'R2': list('D2'),
    'HAL': 9000,
    'K': {'2SO'},
    'C': ['3PO'],
    'BB': '8',
    'L': 3,
    'Poppins': set(list('supercalifragilisticexpialidocious')),
    'Galactica': { 'Karl':'Helo', 'Kara':'Starbuck', 'Lee':'Apollo', 
        'Sharon':'Boomer', 'Marge':'Racetrack', 'Louanne':'Kat' }
}

for var,val in the_vars.items():
    if (var not in dir(stud_module) or
        type(val) != type(getattr(stud_module,var)) or
        val != getattr(stud_module,var)):
            print("Variables incomplete or incorrect.")
            break
else:
    names = ['Germanna_levels','UMW_levels','MaryWash_levels','CNU_levels']
    for name in names:
        globals()[name] = getattr(stud_module,name)
    if (Germanna_levels != UMW_levels and UMW_levels is MaryWash_levels and
        UMW_levels == CNU_levels and UMW_levels is not CNU_levels):
        points += 2
        print("Variables correct! +2")
    else:
        print("Variables incomplete/incorrect.")
    

def test_the_rest():

    global points
    stud_obj = Stud_class()

    if 'plus2' in dir(stud_obj):
        if stud_obj.plus2(27) == 29:
            points += 1
            print("plus2() correct! +1")
        else:    
            print("plus2() incorrect.")
    else:
        print("(plus2() incomplete.)")


    if 'unique_vals' in dir(stud_module):
        uv_func = getattr(stud_module,'unique_vals')
        uv = uv_func({ 'Malone':32, 'Ruth':3, 'Favre':4, 'Jordan':23, 
            'Sandberg':23, 'Kobe':24, 'Jeter':2, 'Brown':32, 'Magic':32, 
            'Elway':7, 'Koufax':32, 'Mantle':7 })
        if (type(uv) == list and len(uv) == 7 and
            all([ x in uv for x in [2,3,4,7,23,24,32]])):
            points += 1
            print("unique_vals() correct! +1")
        else:    
            print("unique_vals() incorrect.")
    else:
        print("(unique_vals() incomplete.)")


    if 'gimme_dat_set' in dir(stud_obj):
        thing1 = stud_obj.gimme_dat_set()
        thing2 = stud_obj.gimme_dat_set()
        if thing1 is thing2 and thing1 == {'Pris','Leon','Roy'}:
            points += 1
            print("gimme_dat_set() correct! +1")
        else:    
            print("gimme_dat_set() incorrect.")
    else:
        print("(gimme_dat_set() incomplete.)")


    if 'gimme_set_like_dat' in dir(stud_obj):
        thing1 = stud_obj.gimme_set_like_dat()
        thing2 = stud_obj.gimme_set_like_dat()
        if (thing1 is not thing2 and thing1 == thing2 and
            thing1 == {'Neo','Morpheus','Trinity'}):
            points += 1
            print("gimme_set_like_dat() correct! +1")
        else:    
            print("gimme_set_like_dat() incorrect.")
    else:
        print("(gimme_set_like_dat() incomplete.)")


    if 'center' in dir(stud_obj):
        if (stud_obj.center('spiderman') == 'e' and
           stud_obj.center('batman') == 'tm' and
           stud_obj.center('') == None and
           stud_obj.center('thor') == 'ho'):
            points += 2
            print("center() correct! +2")
        else:    
            print("center() incorrect.")
    else:
        print("(center() incomplete.)")


    if 'middlest' in dir(stud_module):
        middlest_func = getattr(stud_module,'middlest')
        if (middlest_func(5,9,1) == 5 and
           middlest_func(1,22,3) == 3 and
           middlest_func(1,2,3) == 2 and
           middlest_func(2,2,2) == 2):
            points += 1
            print("middlest() correct! +1")
        else:    
            print("middlest() incorrect.")
        
    else:
        print("(middlest() incomplete.)")


    if 'nuke_last' in dir(stud_obj):
        stuff = list('abcdefedbcaerugioaidsfosa')
        stud_obj.nuke_last(stuff,'o')
        stud_obj.nuke_last(stuff,'a')
        stud_obj.nuke_last(stuff,'s')
        stud_obj.nuke_last(stuff,'e')
        stud_obj.nuke_last(stuff,'r')
        stud_obj.nuke_last(stuff,'i')
        if ''.join(stuff) == 'abcdefedbcaugioadsf':
            points += 2
            print("nuke_last() correct! +2")
        else:    
            print("stuff = {}".format(''.join(stuff)))
            print("nuke_last() incorrect.")
    else:
        print("(nuke_last() incomplete.)")


    if 'tack_on_end' in dir(stud_module):
        tack_on_end_func = getattr(stud_module,'tack_on_end')
        some_thing = ['a','b']
        tack_on_end_func(some_thing,'c')
        tack_on_end_func(some_thing,'d',2)
        tack_on_end_func(some_thing,['e','f'],2)
        tack_on_end_func(some_thing,['g','h','i'])
        if some_thing == ['a','b','c','d','d','e','f','e','f','g','h','i']:
            points += 1
            print("tack_on_end() correct! +1")
        else:    
            print("tack_on_end() incorrect.")
        
    else:
        print("(tack_on_end() incomplete.)")


    if 'wondrous_count' in dir(stud_module):
        wondrous_count_func = getattr(stud_module,'wondrous_count')
        test_vals = { 1:0, 6400:31, 6401:124, 99999:226, 6171:261, 6170:36,
            75128138246:225, 75128138247:1228 }
        if all([wondrous_count_func(tv) == tw for tv,tw in test_vals.items()]):
            points += 2
            print("wondrous_count() correct! +2")
        else:    
            print("wondrous_count() incorrect.")
        
    else:
        print("(wondrous_count() incomplete.)")


    if 'ProbabilityEngine' in dir(stud_module):
        Prob_class = getattr(stud_module,'ProbabilityEngine')
        pe = Prob_class()
        pe2 = Prob_class()
        pe2.add_outcome('blue',300)
        pe2.add_outcome('red',300)
        pe.add_outcome('Alvey',145)
        pe.add_outcome('Arrington',147)
        pe.add_outcome('Ball',105)
        pe.add_outcome('Bushnell',151)
        pe.add_outcome('Custis',42)
        pe.add_outcome('Eagle',624)
        pe.add_outcome('Framar',21)
        pe.add_outcome('Jefferson',100)
        pe.add_outcome('Madison',41)
        pe.add_outcome('Marshall',100)
        pe.add_outcome('Randolph',100)
        pe.add_outcome('Mason',100)
        pe.add_outcome('Russell',173)
        pe.add_outcome('Virginia',183)
        pe.add_outcome('Westmoreland',111)
        pe.add_outcome('Willard',92)
        if (.192 <= pe.compute_prob(['Arrington','Mason','Virginia']) <= .193 and
            .065 <= pe.compute_prob(['Ball','Custis']) <= .066 and
            .279 <= pe.compute_prob(['Eagle']) <= .280 and
            .215 <= pe.compute_prob(['Ball','Custis','Framar',
                                    'Jefferson','Madison','Russell']) <= .216 and
            pe.compute_prob([]) == 0):
            points += 3
            print("ProbabilityEngine part 1 correct! +3")
        else:    
            print("ProbabilityEngine part 1 incorrect. "
                  "({:.4f},{:.4f},{:.4f},{:.4f})".format(
                    pe.compute_prob(['Arrington','Mason','Virginia']),
                    pe.compute_prob(['Ball','Custis']),
                    pe.compute_prob(['Eagle']),
                    pe.compute_prob(['Ball','Custis','Framar',
                                    'Jefferson','Madison','Russell'])))

        students = [ pe.generate_outcome() for _ in range(10000) ]
        colors = [ pe2.generate_outcome() for _ in range(10000) ]
        num_eagle_landing = sum([ x == 'Eagle' for x in students ])
        num_all_girls = sum([ x in ['Ball','Custis'] for x in students ])
        num_no_ac = sum([ x in ['Ball','Custis','Framar',
                    'Jefferson','Madison','Russell'] for x in students ])
        num_upperclass = sum([ x in ['Alvey','Arrington','Ball','Bushnell',
                    'Eagle','Framar','Jefferson','Willard'] for x in students ])
        num_bad_colors = sum([ x in ['blue','red'] for x in students ])
        num_bad_students = sum([ x not in ['blue','red'] for x in colors ])
        if (0 == num_bad_colors and
            0 == num_bad_students and
            2650 <= num_eagle_landing <= 2920 and
            600 <= num_all_girls <= 750 and
            2020 <= num_no_ac <= 2290 and
            6000 <= num_upperclass <= 6400):
            points += 3
            print("ProbabilityEngine part 2 correct! +3")
        else:    
            print("ProbabilityEngine part 2 incorrect. ({},{},{},{})".format(
                num_eagle_landing, num_all_girls, num_no_ac, num_upperclass))
    else:
        print("(ProbabilityEngine incomplete.)")


try:
    Stud_class = getattr(stud_module,sys.argv[1].capitalize() +
        'PythonPractice')
    test_the_rest()
except Exception as err:
    print(str(err))

print("You've earned {}/20 XP!".format(points))
