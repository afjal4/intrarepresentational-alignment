from rdflib.namespace import RDF, RDFS, Namespace

_MN = Namespace("http://metanet.english.ubc.ca/metaphor/MetaphorOntology.owl#")

_type = RDF.type
_label = RDFS.label

_Metaphor = _MN.Metaphor
_Frame = _MN.Frame
_Mapping = _MN.Mapping
_Binding = _MN.Binding
_Example = _MN.Example
_LexicalUnit = _MN.LexicalUnit

_hasName = _MN.hasName
_hasDescription = _MN.hasDescription
_hasStatus = _MN.hasStatus
_hasCulturalScope = _MN.hasCulturalScope
_wasInvestigatedFor = _MN.wasInvestigatedFor

_hasSourceFrame = _MN.hasSourceFrame
_hasTargetFrame = _MN.hasTargetFrame
_hasMappings = _MN.hasMappings
_hasEntailment = _MN.hasEntailment
_hasExample = _MN.hasExample

_isEntailedByMetaphor = _MN.isEntailedByMetaphor
_isTargetSubcaseOfMetaphor = _MN.isTargetSubcaseOfMetaphor
_isSourceSubcaseOfMetaphor = _MN.isSourceSubcaseOfMetaphor
_isInMetaphorFamily = _MN.isInMetaphorFamily

_hasAlias = _MN.hasAlias
_hasMetaphorType = _MN.hasMetaphorType
_hasMetaphorLevel = _MN.hasMetaphorLevel

_hasFrenchCorrespondent = _MN.hasFrenchCorrespondent
_hasSpanishCorrespondent = _MN.hasSpanishCorrespondent

_hasFrameType = _MN.hasFrameType
_hasRoles = _MN.hasRoles
_hasLexicalUnit = _MN.hasLexicalUnit
_hasInference = _MN.hasInference
_hasBindings = _MN.hasBindings
_isInFrameFamily = _MN.isInFrameFamily
_correspondsToFrameNet = _MN.correspondsToFrameNet
_makesUseOfFrame = _MN.makesUseOfFrame

_hasSourceRole = _MN.hasSourceRole
_hasTargetRole = _MN.hasTargetRole

_hasBoundRole1 = _MN.hasBoundRole1
_hasBoundRole2 = _MN.hasBoundRole2

_hasSentence = _MN.hasSentence
_hasAnnotation = _MN.hasAnnotation
_exampleConstruct = _MN["Example.Construction"]
_exampleDialect = _MN["Example.Dialect"]
_isFromLanguage = _MN.isFromLanguage
_hasProvenance = _MN.hasProvenance

_hasLemma = _MN.hasLemma
_LUs_Lemmas = _MN.LUs_Lemmas
_LUs_Language = _MN.LUs_Language

__all__ = [
    "_Binding",
    "_Example",
    "_Frame",
    "_LUs_Language",
    "_LUs_Lemmas",
    "_LexicalUnit",
    "_Mapping",
    "_Metaphor",
    "_correspondsToFrameNet",
    "_exampleConstruct",
    "_exampleDialect",
    "_hasAlias",
    "_hasAnnotation",
    "_hasBindings",
    "_hasBoundRole1",
    "_hasBoundRole2",
    "_hasCulturalScope",
    "_hasDescription",
    "_hasEntailment",
    "_hasExample",
    "_hasFrameType",
    "_hasFrenchCorrespondent",
    "_hasInference",
    "_hasLemma",
    "_hasLexicalUnit",
    "_hasMappings",
    "_hasMetaphorLevel",
    "_hasMetaphorType",
    "_hasName",
    "_hasProvenance",
    "_hasRoles",
    "_hasSentence",
    "_hasSourceFrame",
    "_hasSourceRole",
    "_hasSpanishCorrespondent",
    "_hasStatus",
    "_hasTargetFrame",
    "_hasTargetRole",
    "_isEntailedByMetaphor",
    "_isFromLanguage",
    "_isInFrameFamily",
    "_isInMetaphorFamily",
    "_isSourceSubcaseOfMetaphor",
    "_isTargetSubcaseOfMetaphor",
    "_label",
    "_makesUseOfFrame",
    "_type",
    "_wasInvestigatedFor",
]
