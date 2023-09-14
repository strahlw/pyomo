['c', 'd', 'g', 'h']
2 Var Declarations
    x : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :    -5 :   1.0 :     5 : False : False :  Reals
    y : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   1.0 :     1 : False : False :  Reals

1 Objective Declarations
    obj : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : 100000000.0*x + 1000000.0*y

1 Constraint Declarations
    con : Size=1, Index=None, Active=True
        Key  : Lower : Body  : Upper : Active
        None :   1.0 : x + y :   1.0 :   True

1 Suffix Declarations
    scaling_factor : Direction=Suffix.EXPORT, Datatype=Suffix.FLOAT
        Key : Value
        con :   2.0
        obj : 1e-06
          x :   0.2

5 Declarations: x y obj con scaling_factor
2 Var Declarations
    scaled_x : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  -1.0 :   0.2 :   1.0 : False : False :  Reals
    scaled_y : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :   0.0 :   1.0 :   1.0 : False : False :  Reals

1 Objective Declarations
    scaled_obj : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : 1e-06*(500000000.0*scaled_x + 1000000.0*scaled_y)

1 Constraint Declarations
    scaled_con : Size=1, Index=None, Active=True
        Key  : Lower : Body                          : Upper : Active
        None :   2.0 : 2.0*(5.0*scaled_x + scaled_y) :   2.0 :   True

1 Suffix Declarations
    scaling_factor : Direction=Suffix.EXPORT, Datatype=Suffix.FLOAT
        Key        : Value
        scaled_con :   2.0
        scaled_obj : 1e-06
          scaled_x :   0.2

5 Declarations: scaling_factor scaled_x scaled_y scaled_obj scaled_con
