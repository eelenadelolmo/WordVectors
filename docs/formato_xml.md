****# Formato XML de salida del prototipo

DTD y ejemplos de los XML que conforma las salidas del módulo de selección de rasgos (apartado 6.8.3 de la tesis) y del módulo de identificación de los conjuntos referenciales (apartado 6.8.4 de la tesis).

## 1. Salida del módulo de selección de rasgos

### DTD

```xml
<!DOCTYPE text [
  <!ELEMENT text (sentence+)>
  <!ATTLIST text id CDATA #REQUIRED>
  <!ELEMENT sentence (theme, rheme, semantic_roles)>
  <!ELEMENT theme (token*)>
  <!ELEMENT rheme (token*)>
  <!ELEMENT token (#PCDATA)>
  <!ATTLIST token pos CDATA #REQUIRED>
  <!ELEMENT semantic_roles (frame*)>
  <!ELEMENT frame (argument*)>
  <!ATTLIST frame type CDATA #REQUIRED>
  <!ATTLIST frame head CDATA #REQUIRED>
  <!ELEMENT argument EMPTY>
  <!ATTLIST argument type CDATA #REQUIRED>
  <!ATTLIST argument dependent CDATA #REQUIRED>
]>
```

### Ejemplo

Retomando la descripción técnica del módulo, un ejemplo de la salida que produce es el siguiente —se incluye solamente una de las oraciones del texto porque con ello es suficiente para ejemplificar la estructura del documento, que contiene la secuencia de oraciones de un texto, de tal manera que todas ellas siguen una misma estructura—, en el que se ha seleccionado como Tema de la oración el volumen de Anaya de la serie Nuestro Mundo, uno de los más usados en las aulas vascas y como Rema excede el 55% de temario común fijado por Educación para las comunidades con lengua propia. Como puede observarse, el anotador automático y el generador de correspondencias entre marcos y roles semánticos han localizado un total de cuatro marcos. El primero de ellos, de tipo Aggregate, no tiene ningún argumento y su disparador es comunidades. El segundo de ellos, de tipo Cardinal_numbers, ha sido localizado gracias al disparador uno y contiene un único argumento, de tipo Entity, instanciado en la cadena de los más usados en las aulas vascas. El tercero, de tipo Interior_Profile_Relations, tiene como núcleo en y como argumentos uno de los más usados como Figure y en las aulas vascas como Ground. Finalmente, el cuarto marco, de tipo Text, tiene como núcleo disparador volumen de y como único argumento, de tipo Author, Anaya.

```xml
<text id="1_20000702_ssd_annotated_recursive">
  <sentence>
    <theme>
      el volumen de Anaya de la serie Nuestro Mundo, uno de los más usados en las aulas vascas,
    </theme>
    <rheme>
      excede el 55% de temario común fijado por Educación para las comunidades con lengua propia
    </rheme>
    <semantic_roles>
      <frame type="Aggregate" head="comunidades"></frame>
      <frame type="Cardinal_numbers" head="uno">
        <argument type="Entity" dependent="de los más usados en las aulas vascas"/></frame>
      <frame type="Interior_profile_relation" head="en">
        <argument type="Figure" dependent="uno de los más usados"/>
        <argument type="Ground" dependent="aulas vascas"/></frame>
      <frame type="Text" head="volumen de">
        <argument type="Author" dependent="Anaya"/></frame>
    </semantic_roles>
  </sentence>
</text>
```

### Información morfológica de cada palabra

En la última versión del este módulo, sin embargo, debido a que es muy posible que para futuras mejoras del siguiente módulo, específicamente en la inferencia de las cadenas de correferencia remáticas por su mayor complejidad, se ha persistido la información relativa al etiquetado morfológico de los módulos anteriores. De acuerdo con ello, la descripción del tipo de documento XML descrita anteriormente incluye las declaraciones de los elementos `token` y su atributo `pos` (véase la DTD anterior).

En consonancia con esta nueva gramática propuesta, las etiquetas theme y rheme del ejemplo anterior pasarían a presentar el siguiente formato:

```xml
<sentence>
  <theme>
    <token pos="da0ms0">el</token>
    <token pos="ncms000">volumen</token>
    <token pos="sps00">de</token>
    <token pos="np00000">Anaya</token>
    <token pos="sps00">de</token>
    <token pos="da0fs0">la</token>
    <token pos="ncfs000">serie</token>
    <token pos="dp3cs0">Nuestro</token>
    <token pos="ncfs000">Mundo</token>
    <token pos="fc">,</token>
    <token pos="pi0ms000">uno</token>
    <token pos="sps00">de</token>
    <token pos="da0mp0">los</token>
    <token pos="rg">más</token>
    <token pos="aq0mpp">usados</token>
    <token pos="sps00">en</token>
    <token pos="da0fp0">las</token>
    <token pos="ncfp000">aulas</token>
    <token pos="aq0fp0">vascas</token>
    <token pos="fc">,</token>
  </theme>
  <rheme>
    <token pos="vmip3s0">excede</token>
    <token pos="da0ms0">el</token>
    <token pos="zp">55%</token>
    <token pos="sps00">de</token>
    <token pos="ncms000">temario</token>
    <token pos="aq0cs0">común</token>
    <token pos="aq0msp">fijado</token>
    <token pos="sps00">por</token>
    <token pos="np00000">Educación</token>
    <token pos="sps00">para</token>
    <token pos="da0fp0">las</token>
    <token pos="ncfp000">comunidades</token>
    <token pos="sps00">con</token>
    <token pos="ncfs000">lengua</token>
    <token pos="aq0fs0">propia</token>
  </rheme>
  <semantic_roles>
    [...]
  </semantic_roles>
</sentence>
```

## 2. Salida del módulo de identificación de los conjuntos referenciales

### DTD

De acuerdo con ello, la estructura de este archivo XML final vendría dada por la siguiente DTD, en la que aparecen marcadas con el comentario `<!-- nuevo -->` las declaraciones incluidas en este módulo:

```xml
<!DOCTYPE text [
  <!ELEMENT text (concepts, sentence+)>  <!-- nuevo -->
  <!ATTLIST text id CDATA #REQUIRED>
  <!ELEMENT concepts (concept+)>  <!-- nuevo -->
  <!ELEMENT concept (#PCDATA)>  <!-- nuevo -->
  <!ATTLIST concept id ID #REQUIRED>  <!-- nuevo -->
  <!ELEMENT sentence (theme, rheme, semantic_roles)>
  <!ELEMENT theme (token*)>
  <!ATTLIST theme concept_ref IDREF #REQUIRED>  <!-- nuevo -->
  <!ELEMENT rheme (token*)>
  <!ELEMENT token (#PCDATA)>
  <!ATTLIST token pos CDATA #REQUIRED>
  <!ELEMENT semantic_roles (frame*)>
  <!ELEMENT frame (argument*)>
  <!ATTLIST frame type CDATA #REQUIRED>
  <!ATTLIST frame head CDATA #REQUIRED>
  <!ELEMENT argument EMPTY>
  <!ATTLIST argument type CDATA #REQUIRED>
  <!ATTLIST argument dependent CDATA #REQUIRED>
]>
```

### Ejemplo

Finalmente, un ejemplo de texto anotado en XML de acuerdo con la estructura descrita sería el siguiente:

```xml
<text id="10_20020102_ssd_annotated_recursive">
  <concepts>
    <concept id="A">El Gobierno de Portugal</concept>
    <concept id="B">El ministro</concept>
  </concepts>
  <sentence>
    <theme concept_ref="A">
      <token pos="da0ms0">El</token><token pos="np00000">Gobierno</token><token pos="sps00">de</token><token pos="np00000">Portugal</token>
    </theme>
    <rheme>
      [...]
    </rheme>
    <semantic_roles>
      <frame type="Modality_Reporter" head="el ministro de Economía portugués, d'Oliveira Martins " />
      <frame type="Bringing" head="nacidos el">
        <argument type="Time" dependent="1 de enero" /></frame>
      <frame type="Calendric_unit" head="lunes" />
      <frame type="Closure" head="abrir">
        <argument type="Container_portal" dependent="cuenta de ahorro con 50 euros" /></frame>
      <frame type="Deciding" head="decidió">
        <argument type="Cognizer" dependent="Gobierno de Portugal" />
        <argument type="Decision" dependent="abrir una cuenta de ahorro con 50 euros) a los bebés nacidos el 1 de enero del 2002 ," /></frame>
      <frame type="Leadership" head="Gobierno">
        <argument type="Governed" dependent="de Portugal" /></frame>
      <frame type="People_by_age" head="bebés" />
      <frame type="Spatial_contact" head="a los" />
      <frame type="Statement" head="anunció">
        <argument type="Time" dependent="1 de enero del 2002, según" />
        <argument type="Time" dependent="lunes" /></frame>
      <frame type="Text" head="cuenta">
        <argument type="Author" dependent="con 50 euros" /></frame>
    </semantic_roles>
  </sentence>
  <sentence>
    <theme concept_ref="B">
      <token pos="da0ms0">El</token><token pos="ncms000">ministro</token>
    </theme>
    <rheme>
      [...]
    </rheme>
    <semantic_roles>
      <frame type="Building" head="construcción">
        <argument type="Created_entity" dependent="una Europa de paz y solidaridad" /></frame>
      <frame type="Calendric_unit" head="año" />
      <frame type="Calendric_unit" head="día">
        <argument type="Salient_event" dependent="del año" /></frame>
      <frame type="Collaboration" head="asociando">
        <argument type="Partner_2" dependent="momento histórico" /></frame>
      <frame type="Historic_event" head="histórico" />
      <frame type="Leadership" head="ministro" />
      <frame type="Ordinal_numbers" head="primer">
        <argument type="Type" dependent="día del año" /></frame>
      <frame type="Purpose" head="para">
        <argument type="Goal" dependent="este gesto" />
        <argument type="Goal" dependent="; un momento histórico asociando para siempre a los recién nacidos del primer día del año" /></frame>
    </semantic_roles>
  </sentence>
  <sentence>
    <theme concept_ref="A">
      <token pos="da0ms0">El</token><token pos="np00000">Ejecutivo</token><token pos="aq0ms0">portugués</token>
    </theme>
    <rheme>
      [...]
    </rheme>
    <semantic_roles>
      <frame type="Desiring" head="quería">
        <argument type="Experiencer" dependent="El Ejecutivo portugués" />
        <argument type="Event" dependent="saludar de esta forma el nacimiento de una generación euro" /></frame>
      <frame type="Intentionally_create" head="generación" />
      <frame type="Means" head="forma el">
        <argument type="Purpose" dependent="el nacimiento de una generación euro" /></frame>
    </semantic_roles>
  </sentence>
</text>
```
