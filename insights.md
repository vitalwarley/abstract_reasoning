## objectives

i will analyse the patterns and actions needed to solve some problems
in a given category.

i think one thing really important is to define really well what 
a pattern is e what an action is.

for example: 
  - `shape_mismatch` is the pattern for in-out pairs with different shapes.
  - `object color match` happens when the objects in a given pair stays with the same color.
  - ...
 
for actions, we would have:
  - resize image 
  - resize object 

but each action can have parameters. with `resize image`, for example,
one would have to explicit say how much to scale. it must be learned.

it is important to know the differences between the pairs:
  - image_shape_out - image_shape_in
  - object_shape_out - object_shape_in
  
problems
  - what is the definition of an object?
    - sometimes is straightforward: color continuity, spatial continuity

maybe we don't to be too fine grained... we don't need to know the **exact**
properties of a given pair, but only the **actions**.

for *resize image* we don't need to know the parameters a priori. 

i think this is the right way to do it because most actions only
makes sense chained with other actions, like resize the image and
its objects.

## solving process

### 107
 
#### patterns and actions

- pattern: shape mismatch (10x10 to 20x20)
  - action: resize image shape
- pattern: object color match 
  - action: resize object shape
  
  