组件创建与使用

```
import React from 'react';
class Compounds extends React.Component {

}

导入组件
import Compounds from 路径/Compounds.js

使用
<Compounds/>
```

```
#构造器用于，传递消息
constructor(props) {
    super(props);

    
    
  }
```

{...this.props}
---

> 可以将父组件的所有属性复制给子组件

Object.keys()用于获得由对象属性名组成的数组

```js
classToComponent = {
    ChEMBLCompounds : 'ChEMBLGrid',
    GeneratedMolSet : 'GeneratedGrid',
    SDFCompounds : 'SDFGrid',
    CSVCompounds : 'CSVGrid',
    MolCSVCompounds : 'MolCSVGrid'
  };
var key=Object.keys(classToComponent)

console.log(key)
>>>
['ChEMBLCompounds', 'GeneratedMolSet', 'SDFCompounds', 'CSVCompounds', 'MolCSVCompounds']
```

```js
defaultClass='Molset'
object=key.reduce((object, key) => {
  console.log(key) 
      if (key !== defaultClass) {
        object[key] = classToComponent[key]
      }
      return object
    }, {})
 console.log(object)
>>>
   ChEMBLCompounds
   GeneratedMolSet
   SDFCompounds
   CSVCompounds
   MolCSVCompounds
{ChEMBLCompounds: 'ChEMBLGrid', GeneratedMolSet: 'GeneratedGrid', SDFCompounds: 'SDFGrid', CSVCompounds: 'CSVGrid', MolCSVCompounds: 'MolCSVGrid'} 
```

### hasOwnProperty()

> `hasOwnProperty()` 方法会返回一个布尔值

```

```

