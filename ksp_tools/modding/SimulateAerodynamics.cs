#region Using Statements
using System;
using System.Collections.Generic;
using System.Reflection;
using KRPC.Service;
using KRPC.Service.Attributes;
using KRPC.SpaceCenter.ExtensionMethods; // For ToTuple etc.
using UnityEngine; // Requires reference to UnityEngine.CoreModule.dll
using KSP.UI.Screens; // Requires reference to Assembly-CSharp.dll
using KRPC.SpaceCenter.ExternalAPI;
using KRPC.Utils;
using Tuple3 = System.Tuple<double, double, double>;
using Tuple4 = System.Tuple<double, double, double, double>;
using KRPC.SpaceCenter.Services;
//using System.Linq; // Throws some errors?
using CustomLogging;
#endregion

namespace SimulateAerodynamicForces
{
    [KRPCService(Name = "SimulateAerodynamicForces", GameScene = GameScene.Flight)]
    public static class SimulateAerodynamicForces
    {
        // --- Reflection Cache ---
        // Types
        private static Type _celestialBodyType;
        private static Type _partType;
        private static Type _dragCubeListType;
        private static Type _physicsGlobalsType;
        private static Type _liftingSurfaceCurveType; // Nested in PhysicsGlobals
        private static Type _floatCurveType;
        private static Type _moduleLiftingSurfaceType;
        private static Type _moduleControlSurfaceType;
        private static Type _flightCtrlStateType; // KSP's internal FlightCtrlState
        private static Type _vesselType;
        private static Type _partModuleListType;
        private static Type _attachNodeType;

        // CelestialBody Methods
        private static MethodInfo _cbGetPressureMethod;
        private static MethodInfo _cbGetTemperatureMethod;
        private static MethodInfo _cbGetDensityMethod;
        private static MethodInfo _cbGetSpeedOfSoundMethod;

        // PhysicsGlobals Singleton Instance
        private static PropertyInfo _pgInstanceProp;

        // PhysicsGlobals Instance Fields (accessed via Instance property)
        private static FieldInfo _pgDragMultiplierField;        // dragMultiplier
        private static FieldInfo _pgBodyLiftMultiplierField;    // bodyLiftMultiplier
        private static FieldInfo _pgDragCubeMultiplierField;    // dragCubeMultiplier
        private static FieldInfo _pgLiftMultiplierField;        // liftMultiplier
        private static FieldInfo _pgLiftDragMultiplierField;    // liftDragMultiplier

        // PhysicsGlobals Static Properties
        private static PropertyInfo _pgDragCurvePseudoReynoldsProp;
        private static PropertyInfo _pgBodyLiftCurveProp;      // Returns LiftingSurfaceCurve instance
        private static FieldInfo _pgSurfaceCurvesField;       // Is a static field "SurfaceCurves" of type SurfaceCurvesList (struct)

        // PhysicsGlobals.LiftingSurfaceCurve Fields
        private static FieldInfo _lscLiftCurveField;
        private static FieldInfo _lscLiftMachCurveField;

        // FloatCurve Methods
        private static MethodInfo _fcEvaluateMethod;

        // DragCubeList Properties & Fields
        private static PropertyInfo _dclNoneProp;
        private static FieldInfo _dclFaceDirectionsField;    // static
        private static FieldInfo _dclWeightedAreaField;      // instance
        private static FieldInfo _dclWeightedDragField;      // instance
        private static FieldInfo _dclAreaOccludedField;      // Added for occlusion
        private static FieldInfo _dclSurfaceCurvesField;
        private static FieldInfo _dclBodyLiftCurveField;    // instance, returns PhysicsGlobals.LiftingSurfaceCurve
        private static FieldInfo _dclDragCurveCdField;      // instance
        private static FieldInfo _dclDragCurveCdPowerField; // instance

        // Part Properties & Fields
        private static PropertyInfo _partShieldedProp;
        private static PropertyInfo _partDragCubesProp;
        private static FieldInfo _partTransformField;
        private static FieldInfo _partBodyLiftMultiplierField;
        private static FieldInfo _partDragModelField;
        private static FieldInfo _partMaxDragField;
        private static FieldInfo _partMinDragField;
        private static FieldInfo _partDragReferenceVectorField;
        private static PropertyInfo _partModulesProp;
        private static FieldInfo _partCoLOffsetField;
        private static FieldInfo _partCoPOffsetField;
        private static FieldInfo _partHasLiftModuleField;
        private static MethodInfo _partFindAttachNodeMethod;

        // ModuleLiftingSurface Fields
        private static FieldInfo _mlsBaseTransformField;
        private static FieldInfo _mlsTransformDirField;
        private static FieldInfo _mlsTransformSignField;
        private static FieldInfo _mlsDeflectionLiftCoeffField;
        private static FieldInfo _mlsOmnidirectionalField;
        private static FieldInfo _mlsPerpendicularOnlyField;
        private static FieldInfo _mlsUseInternalDragModelField;
        private static FieldInfo _mlsLiftCurveField;
        private static FieldInfo _mlsLiftMachCurveField;
        private static FieldInfo _mlsDragCurveField;
        private static FieldInfo _mlsDragMachCurveField;
        private static FieldInfo _mlsDisableBodyLiftField;
        private static FieldInfo _mlsDisplaceVelocityField; // Not fully implemented in this sim
        private static FieldInfo _mlsVelocityOffsetField;   // Not fully implemented in this sim
        private static FieldInfo _mlsNodeEnabledField;
        private static FieldInfo _mlsAttachNodeField; // For the protected field
        private static FieldInfo _mlsAttachNodeNameField; // To get the node name if needed for FindAttachNode on Part
        private static FieldInfo _anAttachedPartField;

        // ModuleControlSurface Fields
        private static FieldInfo _mcsCtrlSurfaceAreaField;
        private static FieldInfo _mcsCtrlSurfaceRangeField;
        // private static FieldInfo _mcsActuatorSpeedField; // Not used for instant calculation
        private static FieldInfo _mcsIgnorePitchField;
        private static FieldInfo _mcsIgnoreYawField;
        private static FieldInfo _mcsIgnoreRollField;
        private static FieldInfo _mcsDeployField;
        private static FieldInfo _mcsDeployAngleField;
        private static FieldInfo _mcsDeployInvertField;
        // private static FieldInfo _mcsPartDeployInvertField; // Editor/symmetry specific
        private static FieldInfo _mcsAuthorityLimiterField;
        private static FieldInfo _mcsDeflectionDirectionField;
        // private static FieldInfo _mcsCtrlSurfaceTransformField; // "ctrlSurface" - not directly using its rotation for this simplified deflection
        // private static FieldInfo _mcsNeutralRotationField;      // "neutral" - not directly using its rotation

        // Vessel properties
        private static PropertyInfo _vesselCurrentCoMProp;
        private static PropertyInfo _vesselReferenceTransformProp;
        private static FieldInfo _vesselCtrlStateField; // To get KSP's FlightCtrlState

        // --- State ---
        private static bool _reflectionInitialized = false;
        private static bool _reflectionFailed = false;
        private static readonly BindingFlags _flagsAll = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
        private static readonly BindingFlags _flagsStaticPub = BindingFlags.Static | BindingFlags.Public;
        private static readonly BindingFlags _flagsInstPub = BindingFlags.Instance | BindingFlags.Public;
        private static readonly BindingFlags _flagsInstNonPub = BindingFlags.Instance | BindingFlags.NonPublic;

        private static readonly object _reflectionLock = new object();

        // Logging setup? 
        //CustomLogger.RootContext = "SimAero";

        [KRPCClass()]
        public class SimAeroFlightControlStateInput
        {
            [KRPCProperty]
            public float Pitch { get; set; } = 0f;
            [KRPCProperty]
            public float Yaw { get; set; } = 0f;
            [KRPCProperty]
            public float Roll { get; set; } = 0f;
        }

        [KRPCProcedure()]
        public static bool EnsureReflectionInitialized()
        {
            if (_reflectionInitialized && !_reflectionFailed) return true;
            lock (_reflectionLock)
            {
                if (_reflectionInitialized && !_reflectionFailed) return true;
                _reflectionInitialized = true;
                _reflectionFailed = false;
                CustomLogger.Log("Starting Reflection Initializaion...");
                try
                {
                    _celestialBodyType = typeof(CelestialBody);
                    _partType = typeof(Part);
                    _dragCubeListType = typeof(DragCubeList);
                    _physicsGlobalsType = typeof(PhysicsGlobals);
                    _liftingSurfaceCurveType = _physicsGlobalsType.GetNestedType("LiftingSurfaceCurve", _flagsAll);
                    _floatCurveType = typeof(FloatCurve);
                    _moduleLiftingSurfaceType = typeof(ModuleLiftingSurface);
                    _moduleControlSurfaceType = typeof(ModuleControlSurface);
                    _flightCtrlStateType = typeof(FlightCtrlState);
                    _vesselType = typeof(Vessel);
                    //_partModuleListType = typeof(IPartModuleList);

                    Action<object, string, string> throwIfNull = (obj, typeName, memberName) => { if (obj == null) throw new MissingMemberException(typeName, memberName); };

                    // CelestialBody
                    _cbGetPressureMethod = _celestialBodyType.GetMethod("GetPressure", _flagsAll, null, new Type[] { typeof(double) }, null); throwIfNull(_cbGetPressureMethod, "CB", "GetPressure");
                    _cbGetTemperatureMethod = _celestialBodyType.GetMethod("GetFullTemperature", _flagsAll, null, new Type[] { typeof(double), typeof(double) }, null); throwIfNull(_cbGetTemperatureMethod, "CB", "GetFullTemp");
                    _cbGetDensityMethod = _celestialBodyType.GetMethod("GetDensity", _flagsAll, null, new Type[] { typeof(double), typeof(double) }, null); throwIfNull(_cbGetDensityMethod, "CB", "GetDensity");
                    _cbGetSpeedOfSoundMethod = _celestialBodyType.GetMethod("GetSpeedOfSound", _flagsAll, null, new Type[] { typeof(double), typeof(double) }, null); throwIfNull(_cbGetSpeedOfSoundMethod, "CB", "GetSoS");

                    // PhysicsGlobals
                    _pgInstanceProp = _physicsGlobalsType.GetProperty("Instance", _flagsStaticPub); throwIfNull(_pgInstanceProp, "PG", "Instance");
                    _pgDragMultiplierField = _physicsGlobalsType.GetField("dragMultiplier", _flagsInstNonPub); throwIfNull(_pgDragMultiplierField, "PG", "dragMultiplier");
                    _pgBodyLiftMultiplierField = _physicsGlobalsType.GetField("bodyLiftMultiplier", _flagsInstNonPub); throwIfNull(_pgBodyLiftMultiplierField, "PG", "bodyLiftMultiplier");
                    _pgDragCubeMultiplierField = _physicsGlobalsType.GetField("dragCubeMultiplier", _flagsInstNonPub); throwIfNull(_pgDragCubeMultiplierField, "PG", "dragCubeMultiplier");
                    _pgLiftMultiplierField = _physicsGlobalsType.GetField("liftMultiplier", _flagsInstNonPub); throwIfNull(_pgLiftMultiplierField, "PG", "liftMultiplier");
                    _pgLiftDragMultiplierField = _physicsGlobalsType.GetField("liftDragMultiplier", _flagsInstNonPub); throwIfNull(_pgLiftDragMultiplierField, "PG", "liftDragMultiplier");
                    _pgDragCurvePseudoReynoldsProp = _physicsGlobalsType.GetProperty("DragCurvePseudoReynolds", _flagsStaticPub); throwIfNull(_pgDragCurvePseudoReynoldsProp, "PG", "DragCurvePseudoReynolds");
                    _pgBodyLiftCurveProp = _physicsGlobalsType.GetProperty("BodyLiftCurve", _flagsStaticPub); throwIfNull(_pgBodyLiftCurveProp, "PG", "BodyLiftCurve");
                    _pgSurfaceCurvesField = _physicsGlobalsType.GetField("SurfaceCurves", _flagsStaticPub); throwIfNull(_pgSurfaceCurvesField, "PG", "SurfaceCurves"); // It's a static field

                    // PhysicsGlobals.LiftingSurfaceCurve
                    throwIfNull(_liftingSurfaceCurveType, "PG", "LiftingSurfaceCurveType");
                    _lscLiftCurveField = _liftingSurfaceCurveType.GetField("liftCurve", _flagsInstPub); throwIfNull(_lscLiftCurveField, "PG.LSC", "liftCurve");
                    _lscLiftMachCurveField = _liftingSurfaceCurveType.GetField("liftMachCurve", _flagsInstPub); throwIfNull(_lscLiftMachCurveField, "PG.LSC", "liftMachCurve");

                    // FloatCurve
                    _fcEvaluateMethod = _floatCurveType.GetMethod("Evaluate", _flagsInstPub, null, new Type[] { typeof(float) }, null); throwIfNull(_fcEvaluateMethod, "FloatCurve", "Evaluate");

                    // DragCubeList
                    _dclNoneProp = _dragCubeListType.GetProperty("None", _flagsInstPub); throwIfNull(_dclNoneProp, "DCL", "None");
                    _dclFaceDirectionsField = _dragCubeListType.GetField("faceDirections", _flagsAll); throwIfNull(_dclFaceDirectionsField, "DCL", "faceDirections"); // static
                    _dclWeightedAreaField = _dragCubeListType.GetField("weightedArea", _flagsInstNonPub); throwIfNull(_dclWeightedAreaField, "DCL", "weightedArea");
                    _dclWeightedDragField = _dragCubeListType.GetField("weightedDrag", _flagsInstNonPub); throwIfNull(_dclWeightedDragField, "DCL", "weightedDrag");
                    _dclAreaOccludedField = _dragCubeListType.GetField("areaOccluded", _flagsInstNonPub); throwIfNull(_dclAreaOccludedField, "DCL", "areaOccluded");
                    _dclSurfaceCurvesField = _dragCubeListType.GetField("SurfaceCurves", _flagsInstPub); throwIfNull(_dclSurfaceCurvesField, "DCL", "SurfaceCurves");
                    _dclBodyLiftCurveField = _dragCubeListType.GetField("BodyLiftCurve", _flagsInstPub); throwIfNull(_dclBodyLiftCurveField, "DCL", "BodyLiftCurve");
                    _dclDragCurveCdField = _dragCubeListType.GetField("DragCurveCd", _flagsInstPub); throwIfNull(_dclDragCurveCdField, "DCL", "DragCurveCd");
                    _dclDragCurveCdPowerField = _dragCubeListType.GetField("DragCurveCdPower", _flagsInstPub); throwIfNull(_dclDragCurveCdPowerField, "DCL", "DragCurveCdPower");

                    // Part
                    _partShieldedProp = _partType.GetProperty("ShieldedFromAirstream", _flagsInstPub); throwIfNull(_partShieldedProp, "Part", "Shielded");
                    _partDragCubesProp = _partType.GetProperty("DragCubes", _flagsInstPub); throwIfNull(_partDragCubesProp, "Part", "DragCubes");
                    _partTransformField = _partType.GetField("partTransform", _flagsInstPub); throwIfNull(_partTransformField, "Part", "partTransform");
                    _partBodyLiftMultiplierField = _partType.GetField("bodyLiftMultiplier", _flagsInstPub); throwIfNull(_partBodyLiftMultiplierField, "Part", "bodyLiftMultiplier");
                    _partDragModelField = _partType.GetField("dragModel", _flagsInstPub); throwIfNull(_partDragModelField, "Part", "dragModel");
                    _partMaxDragField = _partType.GetField("maximum_drag", _flagsInstPub); throwIfNull(_partMaxDragField, "Part", "max_drag");
                    _partMinDragField = _partType.GetField("minimum_drag", _flagsInstPub); throwIfNull(_partMinDragField, "Part", "min_drag");
                    _partDragReferenceVectorField = _partType.GetField("dragReferenceVector", _flagsInstPub); throwIfNull(_partDragReferenceVectorField, "Part", "dragRefVec");
                    _partModulesProp = _partType.GetProperty("Modules", _flagsInstPub); throwIfNull(_partModulesProp, "Part", "Modules");
                    _partCoLOffsetField = _partType.GetField("CoLOffset"); throwIfNull(_partCoLOffsetField, "Part", "CoLOffset");
                    _partCoPOffsetField = _partType.GetField("CoPOffset"); throwIfNull(_partCoPOffsetField, "Part", "CoPOffset");
                    _partHasLiftModuleField = _partType.GetField("hasLiftModule"); throwIfNull(_partHasLiftModuleField, "Part", "hasLiftModule");

                    // ModuleLiftingSurface
                    _mlsBaseTransformField = _moduleLiftingSurfaceType.GetField("baseTransform", _flagsInstNonPub); throwIfNull(_mlsBaseTransformField, "MLS", "baseTransform");
                    _mlsTransformDirField = _moduleLiftingSurfaceType.GetField("transformDir", _flagsInstPub); throwIfNull(_mlsTransformDirField, "MLS", "transformDir");
                    _mlsTransformSignField = _moduleLiftingSurfaceType.GetField("transformSign", _flagsInstPub); throwIfNull(_mlsTransformSignField, "MLS", "transformSign");
                    _mlsDeflectionLiftCoeffField = _moduleLiftingSurfaceType.GetField("deflectionLiftCoeff", _flagsInstPub); throwIfNull(_mlsDeflectionLiftCoeffField, "MLS", "deflectionLiftCoeff");
                    _mlsOmnidirectionalField = _moduleLiftingSurfaceType.GetField("omnidirectional", _flagsInstPub); throwIfNull(_mlsOmnidirectionalField, "MLS", "omnidirectional");
                    _mlsPerpendicularOnlyField = _moduleLiftingSurfaceType.GetField("perpendicularOnly", _flagsInstPub); throwIfNull(_mlsPerpendicularOnlyField, "MLS", "perpendicularOnly");
                    _mlsUseInternalDragModelField = _moduleLiftingSurfaceType.GetField("useInternalDragModel", _flagsInstPub); throwIfNull(_mlsUseInternalDragModelField, "MLS", "useInternalDragModel");
                    _mlsLiftCurveField = _moduleLiftingSurfaceType.GetField("liftCurve", _flagsInstPub); throwIfNull(_mlsLiftCurveField, "MLS", "liftCurve");
                    _mlsLiftMachCurveField = _moduleLiftingSurfaceType.GetField("liftMachCurve", _flagsInstPub); throwIfNull(_mlsLiftMachCurveField, "MLS", "liftMachCurve");
                    _mlsDragCurveField = _moduleLiftingSurfaceType.GetField("dragCurve", _flagsInstPub); throwIfNull(_mlsDragCurveField, "MLS", "dragCurve");
                    _mlsDragMachCurveField = _moduleLiftingSurfaceType.GetField("dragMachCurve", _flagsInstPub); throwIfNull(_mlsDragMachCurveField, "MLS", "dragMachCurve");
                    _mlsDisableBodyLiftField = _moduleLiftingSurfaceType.GetField("disableBodyLift", _flagsInstPub); throwIfNull(_mlsDisableBodyLiftField, "MLS", "disableBodyLift");
                    _mlsDisplaceVelocityField = _moduleLiftingSurfaceType.GetField("displaceVelocity", _flagsInstPub); throwIfNull(_mlsDisplaceVelocityField, "MLS", "displaceVelocity");
                    _mlsVelocityOffsetField = _moduleLiftingSurfaceType.GetField("velocityOffset", _flagsInstPub); throwIfNull(_mlsVelocityOffsetField, "MLS", "velocityOffset");

                    // ModuleControlSurface
                    _mcsCtrlSurfaceAreaField = _moduleControlSurfaceType.GetField("ctrlSurfaceArea", _flagsInstPub); throwIfNull(_mcsCtrlSurfaceAreaField, "MCS", "ctrlSurfaceArea");
                    _mcsCtrlSurfaceRangeField = _moduleControlSurfaceType.GetField("ctrlSurfaceRange", _flagsInstPub); throwIfNull(_mcsCtrlSurfaceRangeField, "MCS", "ctrlSurfaceRange");
                    _mcsIgnorePitchField = _moduleControlSurfaceType.GetField("ignorePitch", _flagsInstPub); throwIfNull(_mcsIgnorePitchField, "MCS", "ignorePitch");
                    _mcsIgnoreYawField = _moduleControlSurfaceType.GetField("ignoreYaw", _flagsInstPub); throwIfNull(_mcsIgnoreYawField, "MCS", "ignoreYaw");
                    _mcsIgnoreRollField = _moduleControlSurfaceType.GetField("ignoreRoll", _flagsInstPub); throwIfNull(_mcsIgnoreRollField, "MCS", "ignoreRoll");
                    _mcsDeployField = _moduleControlSurfaceType.GetField("deploy", _flagsInstPub); throwIfNull(_mcsDeployField, "MCS", "deploy");
                    _mcsDeployAngleField = _moduleControlSurfaceType.GetField("deployAngle", _flagsInstPub); throwIfNull(_mcsDeployAngleField, "MCS", "deployAngle");
                    _mcsDeployInvertField = _moduleControlSurfaceType.GetField("deployInvert", _flagsInstPub); throwIfNull(_mcsDeployInvertField, "MCS", "deployInvert");
                    _mcsAuthorityLimiterField = _moduleControlSurfaceType.GetField("authorityLimiter", _flagsInstPub); throwIfNull(_mcsAuthorityLimiterField, "MCS", "authorityLimiter");
                    _mcsDeflectionDirectionField = _moduleControlSurfaceType.GetField("deflectionDirection", _flagsAll); throwIfNull(_mcsDeflectionDirectionField, "MCS", "deflectionDirection");

                    // Vessel
                    _vesselCurrentCoMProp = _vesselType.GetProperty("CurrentCoM", _flagsInstPub); throwIfNull(_vesselCurrentCoMProp, "Vessel", "CurrentCoM");
                    _vesselReferenceTransformProp = _vesselType.GetProperty("ReferenceTransform", _flagsInstPub); throwIfNull(_vesselReferenceTransformProp, "Vessel", "ReferenceTransform");
                    _vesselCtrlStateField = _vesselType.GetField("ctrlState", _flagsInstPub); throwIfNull(_vesselCtrlStateField, "Vessel", "ctrlState");

                    // AttachNode stuff
                    _attachNodeType = typeof(AttachNode); // Assuming AttachNode is directly accessible by type
                                                          // If not, you might need:
                                                          // Assembly assemblyCSharp = Assembly.Load("Assembly-CSharp");
                                                          // _attachNodeType = assemblyCSharp.GetType("AttachNode");
                    throwIfNull(_attachNodeType, "AttachNode", "Type");

                    // ModuleLiftingSurface fields for node logic
                    _mlsNodeEnabledField = _moduleLiftingSurfaceType.GetField("nodeEnabled", _flagsInstPub);
                    throwIfNull(_mlsNodeEnabledField, "MLS", "nodeEnabled");

                    _mlsAttachNodeField = _moduleLiftingSurfaceType.GetField("attachNode", _flagsInstNonPub); // Use BindingFlags.NonPublic for protected
                    throwIfNull(_mlsAttachNodeField, "MLS", "attachNode (protected)");

                    _mlsAttachNodeNameField = _moduleLiftingSurfaceType.GetField("attachNodeName", _flagsInstPub); // Public KSPField
                    throwIfNull(_mlsAttachNodeNameField, "MLS", "attachNodeName");

                    // AttachNode fields
                    _anAttachedPartField = _attachNodeType.GetField("attachedPart", _flagsInstPub);
                    throwIfNull(_anAttachedPartField, "AttachNode", "attachedPart");

                    // Part method (if you choose to re-fetch AttachNode via Part.FindAttachNode)
                    _partFindAttachNodeMethod = _partType.GetMethod("FindAttachNode", _flagsInstPub, null, new Type[] { typeof(string) }, null);
                    throwIfNull(_partFindAttachNodeMethod, "Part", "FindAttachNode");

                    CustomLogger.Log("Reflection initialzied successfully");
                }
                catch (Exception e)
                {
                    CustomLogger.LogError("Reflection initialization FAILED", e);
                    _reflectionFailed = true;
                    throw new InvalidOperationException("Reflection failed, cannot proceed.", e);
                }
            }
            return !_reflectionFailed;
        }

        // Main Function
        [KRPCProcedure()]
        public static Tuple<double, double, double> CalculateVesselAeroForce(
            Tuple<double, double, double> relPosition,
            Tuple<double, double, double> relVelocity,
            ReferenceFrame referenceFrame,
            bool useMLS = true,
            bool useMCS = true,
            bool initialize = true,
            float MLSMult = 1000f,
            float pitch = 0f,
            float yaw = 0f,
            float roll = 0f,
            bool verbose = true)
        {
            CustomLogger.RootContext = "SimAero";
            CustomLogger.NumericPrecision = 4;
            CustomLogger.IsVerbose = verbose;
            CustomLogger.EnterFunction("CalcAeroForce");
            CustomLogger.Log("Starting Calculation");

            if (initialize)
            {
                _reflectionFailed = false;
                _reflectionInitialized = false;
            }

            CustomLogger.EnterFunction("ReflectInit");
            if (!EnsureReflectionInitialized())
            {
                throw new InvalidOperationException("Reflection initialization failed. Cannot calculate aerodynamics.");
            }
            CustomLogger.ExitFunction();

            SimAeroFlightControlStateInput controlInputs = new SimAeroFlightControlStateInput // Set all zero right now.
            {
                Pitch = pitch,
                Yaw = yaw,
                Roll = roll
            };

            Vessel vessel = FlightGlobals.ActiveVessel;
            if (vessel == null) throw new InvalidOperationException("No active vessel found.");
            CelestialBody body = vessel.mainBody;
            if (body == null) throw new InvalidOperationException("Active vessel has no main body.");

            var worldVelocity = referenceFrame.VelocityToWorldSpace(relPosition.ToVector(), relVelocity.ToVector());
            var worldPosition = referenceFrame.PositionToWorldSpace(relPosition.ToVector());
            // UnityEngine.Debug.Log($"[SimulateAeroServices] worldPosition: {worldPosition}, worldVelocity: {worldVelocity}");
            CustomLogger.Log("State", "worldPos", worldPosition, "worldVel", worldVelocity);

            Vector3d posVec = new Vector3d(worldPosition.x, worldPosition.y, worldPosition.z);
            Vector3 velVec = new Vector3d(worldVelocity.x, worldVelocity.y, worldVelocity.z);
            float speedSqr = velVec.sqrMagnitude;
            float speed = (speedSqr > 1e-9f) ? Mathf.Sqrt(speedSqr) : 0f;
            Vector3 worldVelDir = (speed > 1e-6f) ? (velVec / speed) : Vector3.zero;
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Speed: {speed:F4}, ");
            CustomLogger.Log("Speed", speed);

            CustomLogger.EnterFunction("GetEnviron");
            EnvironmentalData environment = GetEnvironmentalConditions(posVec, body);
            CustomLogger.ExitFunction();

            if (environment.Density <= 0) return new Tuple<double, double, double>(0, 0, 0);

            float mach = (environment.SpeedOfSound > 1e-6) ? (speed / (float)environment.SpeedOfSound) : 0f;
            double dynamicPressurePa_Value = 0.5 * environment.Density * speedSqr;
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Environment: " +
            // $"pressure (kPa): {environment.PressureKPa:F4}, " +
            // $"temperature: {environment.Temperature:F4}, " +
            // $"density: {environment.Density:F4}, " +
            // $"speed of sound: {environment.SpeedOfSound:F4}");
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Mach: {mach}, dynamic pressure (Pa): {dynamicPressurePa_Value}.");
            CustomLogger.Log("environment", "Pressure (kPa)", environment.PressureKPa,
                            "Temperature (K)", environment.Temperature, "Density (kg/m3)", environment.Density,
                            "SpeedOfSound (m/s)", environment.SpeedOfSound);
            CustomLogger.Log("Mach", mach);
            CustomLogger.Log("Dynamic Pressure (Pa)", dynamicPressurePa_Value);

            CustomLogger.EnterFunction("GetGlobals");
            GlobalAeroModifiers globals = GetGlobalModifiers(mach, (float)environment.Density, speed);
            CustomLogger.ExitFunction();

            CustomLogger.Log("globals", "DragMultiplier", globals.DragMultiplier,
                            "BodyLiftMultiplier", globals.BodyLiftMultiplier,
                            "PG.DragCubeMultplier", globals.DragCubeMultiplier_PG,
                            "LiftMultiplier", globals.LiftMultiplier,
                            "LiftDragMultiplier", globals.LiftDragMultiplier,
                            "PseudoReDragMult", globals.PseudoReDragMult);

            Vector3 totalForceOnVessel = Vector3.zero;
            Vector3 totalBodyDragForce = Vector3.zero;
            Vector3 totalBodyLiftForce = Vector3.zero;
            Vector3 totalMLSDragForce = Vector3.zero;
            Vector3 totalMLSLiftForce = Vector3.zero;

            Vector3 vesselCurrentCoM = (Vector3)_vesselCurrentCoMProp.GetValue(vessel, null);
            Transform vesselReferenceTransform = (Transform)_vesselReferenceTransformProp.GetValue(vessel, null);
            // FlightCtrlState actualVesselFlightCtrlState = (FlightCtrlState)_vesselCtrlStateProp.GetValue(vessel, null);

            // UnityEngine.Debug.Log($"\n[SimulateAeroServices] Entering part loop.");
            CustomLogger.Log("Entering part loop");
            CustomLogger.EnterFunction("PartLoop");
            foreach (Part part in vessel.parts)
            {
                Vector3 partNetAeroForce = Vector3.zero;
                Vector3 baseDragForce = Vector3.zero;
                Vector3 baseBodyLiftForce = Vector3.zero;

                CustomLogger.EnterFunction("CalcBodyForces");
                CalculatePartAerodynamics_BaseBodyForces(part, worldVelDir, mach, (float)dynamicPressurePa_Value, environment, globals,
                                                 out baseDragForce, out baseBodyLiftForce);
                CustomLogger.Log("baseDragForce", baseDragForce);
                CustomLogger.Log("baseBodyLiftForce", baseBodyLiftForce);
                CustomLogger.ExitFunction();

                Vector3 wingLiftForce = Vector3.zero;
                Vector3 wingDragForce = Vector3.zero;

                //PartModule foundModule = null; // Can be MLS or MCS

                bool mcsFound = false;
                PartModule mcsModule = null;
                PartModule mlsModule = null; // Will hold either MCS or MLS if found

                PartModuleList partModules = (PartModuleList)_partModulesProp.GetValue(part, null);
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Modules fetched from part");
                CustomLogger.Log("Modules fetched from part");

                for (int i = 0; i < partModules.Count; i++)
                {
                    PartModule pm = partModules[i];
                    if (pm.GetType() == _moduleControlSurfaceType)
                    {
                        // UnityEngine.Debug.Log($"[SimulateAeroServices] part has Module of type ModuleControlSurface");
                        CustomLogger.Log("Module is type ModuleControlSurface");
                        mlsModule = pm; // MCS is an MLS, store it in mlsModule for flags later
                        mcsModule = pm; // Store specifically as MCS for its own logic
                        mcsFound = true;
                        break;
                    }
                }

                if (!mcsFound)
                {
                    for (int i = 0; i < partModules.Count; i++)
                    {
                        PartModule pm = partModules[i];
                        if (pm.GetType() == _moduleLiftingSurfaceType)
                        {
                            // UnityEngine.Debug.Log($"[SimulateAeroServices] part has Module of type ModuleLiftingSurface");
                            CustomLogger.Log("Module is type ModuleLiftingSurface");
                            mlsModule = pm;
                            break;
                        }
                    }
                }

                bool disableBodyLiftForPart = false;

                if (mcsFound && useMCS)
                {
                    CustomLogger.Log("ModuleControlSurface found");
                    CustomLogger.EnterFunction("CalcMCS");
                    CalculateModuleControlSurfaceForce((ModuleControlSurface)mcsModule, part, worldVelDir, speed, mach,
                                                       (float)dynamicPressurePa_Value, environment, globals,
                                                       controlInputs, vesselCurrentCoM, vesselReferenceTransform,
                                                       out wingLiftForce, out wingDragForce);
                    CustomLogger.Log("wingLiftForce", wingLiftForce);
                    CustomLogger.Log("wingDragForce", wingDragForce);
                    CustomLogger.ExitFunction();

                    disableBodyLiftForPart = (bool)_mlsDisableBodyLiftField.GetValue(mcsModule); // MCS inherits from MLS
                    CustomLogger.Log("disableBodyLiftForPart", disableBodyLiftForPart);

                }
                else if ((mlsModule != null) && useMLS) // It's an MLS
                {
                    CustomLogger.Log("ModuleLiftingSurface found");
                    CustomLogger.EnterFunction("CalcMLS");
                    CalculateModuleLiftingSurfaceForce((ModuleLiftingSurface)mlsModule, part, worldVelDir, speed, mach,
                                                       (float)dynamicPressurePa_Value, environment, globals,
                                                       out wingLiftForce, out wingDragForce);
                    CustomLogger.Log("wingLiftForce", wingLiftForce);
                    CustomLogger.Log("wingDragForce", wingDragForce);
                    CustomLogger.ExitFunction();

                    disableBodyLiftForPart = (bool)_mlsDisableBodyLiftField.GetValue(mlsModule);
                    CustomLogger.Log("disableBodyLiftForPart", disableBodyLiftForPart);
                }

                // Combine forces for the part
                partNetAeroForce += baseDragForce; // Always add base drag from cubes/simple models
                totalBodyDragForce += baseDragForce;
                bool hasLiftModule = (bool)_partHasLiftModuleField.GetValue(part);
                if (!disableBodyLiftForPart && !hasLiftModule)
                {
                    CustomLogger.Log("Body lift is not disabled, including it");
                    partNetAeroForce += baseBodyLiftForce;
                    totalBodyLiftForce += baseBodyLiftForce;
                }
                else
                {
                    CustomLogger.Log("Body lift is disabled");
                }
                partNetAeroForce += wingLiftForce * MLSMult; // Add lift from MLS/MCS
                partNetAeroForce += wingDragForce * MLSMult; // Add drag from MLS/MCS (if useInternalDragModel is true)
                totalMLSLiftForce += wingLiftForce * MLSMult;
                totalMLSDragForce += wingDragForce * MLSMult;

                totalForceOnVessel += partNetAeroForce;
            }
            CustomLogger.ExitFunction();
            //return totalForceOnVessel.ToTuple();
            CustomLogger.Log("Total body drag force", totalBodyDragForce.magnitude);
            CustomLogger.Log("Total body lift force", totalBodyLiftForce.magnitude);
            CustomLogger.Log("Total ModuleLiftingSurface drag force", totalMLSDragForce.magnitude);
            CustomLogger.Log("Total ModuleLiftingSurface lift force", totalMLSLiftForce.magnitude);
            CustomLogger.Log("Total drag force", (totalBodyDragForce + totalMLSDragForce).magnitude);
            CustomLogger.Log("Total lift force", (totalBodyLiftForce + totalMLSLiftForce).magnitude);
            CustomLogger.Log("Total force before transform", totalForceOnVessel);
            CustomLogger.Log("Magnitude", totalForceOnVessel.magnitude);

            CustomLogger.Log("Ending calculation");
            CustomLogger.ExitFunction();

            return FromWorldSpace(totalForceOnVessel, referenceFrame);
        }

        // Helper function
        private static Tuple3 FromWorldSpace(Vector3 vector, ReferenceFrame refframe)
        {
            return refframe.DirectionFromWorldSpace(vector).ToTuple();
        }

        private struct EnvironmentalData
        {
            public double PressureKPa;
            public double Temperature;
            public double Density;
            public double SpeedOfSound;
        }

        private struct GlobalAeroModifiers
        {
            public float DragMultiplier;          // PhysicsGlobals.DragMultiplier
            public float BodyLiftMultiplier;      // PhysicsGlobals.BodyLiftMultiplier
            public float DragCubeMultiplier_PG;   // PhysicsGlobals.DragCubeMultiplier
            public float LiftMultiplier;          // PhysicsGlobals.LiftMultiplier (for wings)
            public float LiftDragMultiplier;      // PhysicsGlobals.LiftDragMultiplier (for wings)
            public float PseudoReDragMult;
            public object BodyLiftCurveInstance;  // Instance of PhysicsGlobals.LiftingSurfaceCurve for body lift
            public object SurfaceCurvesInstance;  // Instance of PhysicsGlobals.SurfaceCurvesList (struct)
        }

        // Helper function
        private static EnvironmentalData GetEnvironmentalConditions(Vector3d worldPosition, CelestialBody body)
        {
            EnvironmentalData data = new EnvironmentalData();
            double altitude = FlightGlobals.getAltitudeAtPos(worldPosition, body);

            data.PressureKPa = (double)_cbGetPressureMethod.Invoke(body, new object[] { altitude });
            data.Temperature = (double)_cbGetTemperatureMethod.Invoke(body, new object[] { altitude, 0.0 });

            if (data.PressureKPa <= 0 || data.Temperature <= 0)
            {
                data.Density = 0;
                data.SpeedOfSound = 0;
            }
            else
            {
                data.Density = (double)_cbGetDensityMethod.Invoke(body, new object[] { data.PressureKPa, data.Temperature });
                data.SpeedOfSound = (double)_cbGetSpeedOfSoundMethod.Invoke(body, new object[] { data.PressureKPa, data.Density });
            }
            if (data.Density < 0) data.Density = 0;
            return data;
        }

        // Helper function
        private static GlobalAeroModifiers GetGlobalModifiers(float mach, float density, float speed)
        {
            GlobalAeroModifiers mods = new GlobalAeroModifiers();
            object pgInstance = _pgInstanceProp.GetValue(null, null);

            mods.DragMultiplier = (float)_pgDragMultiplierField.GetValue(pgInstance);
            mods.BodyLiftMultiplier = (float)_pgBodyLiftMultiplierField.GetValue(pgInstance);
            mods.DragCubeMultiplier_PG = (float)_pgDragCubeMultiplierField.GetValue(pgInstance);
            mods.LiftMultiplier = (float)_pgLiftMultiplierField.GetValue(pgInstance);
            mods.LiftDragMultiplier = (float)_pgLiftDragMultiplierField.GetValue(pgInstance);

            FloatCurve pseudoReynoldsCurve = (FloatCurve)_pgDragCurvePseudoReynoldsProp.GetValue(null, null); // Static property
            float pseudoReynolds = density * speed;
            mods.PseudoReDragMult = (float)_fcEvaluateMethod.Invoke(pseudoReynoldsCurve, new object[] { pseudoReynolds });

            mods.BodyLiftCurveInstance = _pgBodyLiftCurveProp.GetValue(null, null); // Static property
            mods.SurfaceCurvesInstance = _pgSurfaceCurvesField.GetValue(null); // Static field
            return mods;
        }

        // Helper function
        private static void CalculatePartAerodynamics_BaseBodyForces(Part part, Vector3 worldVelDir, float mach, float dynamicPressurePa,
                                                              EnvironmentalData environment, GlobalAeroModifiers globals,
                                                              out Vector3 dragForce, out Vector3 bodyLiftForce)
        {
            dragForce = Vector3.zero;
            bodyLiftForce = Vector3.zero;

            bool shielded = (bool)_partShieldedProp.GetValue(part, null);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Shielded: {shielded}");
            CustomLogger.Log("Part shielded", shielded);
            if (shielded) return;

            Part.DragModel dragModel = (Part.DragModel)_partDragModelField.GetValue(part);
            Transform partTransform = (Transform)_partTransformField.GetValue(part);
            Vector3 localVelocityDir = (worldVelDir == Vector3.zero) ? Vector3.zero : partTransform.InverseTransformDirection(worldVelDir);

            if (dragModel == Part.DragModel.CUBE)
            {
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Using CUBE DragModel");
                CustomLogger.Log("Using CUBE DragModel");
                object dragCubesObj = _partDragCubesProp.GetValue(part, null);
                if (!(bool)_dclNoneProp.GetValue(dragCubesObj, null))
                {
                    // UnityEngine.Debug.Log($"[SimulateAeroServices] Entering CalculateDragCubeForces()");
                    CustomLogger.EnterFunction("CalcDragCube");
                    CalculateDragCubeForces(dragCubesObj, part, localVelocityDir, mach, dynamicPressurePa, environment, globals, partTransform, out dragForce, out bodyLiftForce);
                    CustomLogger.ExitFunction();
                    // UnityEngine.Debug.Log($"[SimulateAeroServices] Exiting CalculateDragCubeForces()");
                }
            }
            else if (dragModel != Part.DragModel.NONE)
            {
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Using Simple DragModel");
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Entering CalculateSimpleDragValue()");
                CustomLogger.Log("Using simple DragModel");
                CustomLogger.EnterFunction("CalcSimpleDrag");
                double dragValue = CalculateSimpleDragValue(part, dragModel, localVelocityDir);
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Exiting CalculateSimpleDragValue()");
                CustomLogger.ExitFunction();

                float dragMagnitudeN = (float)(dynamicPressurePa * dragValue * globals.PseudoReDragMult * globals.DragMultiplier);
                // UnityEngine.Debug.Log($"[SimulateAeroServices] dragMagnitudeN: {dragMagnitudeN:F4}");
                CustomLogger.Log("dragMagnitudeN", dragMagnitudeN);

                dragForce = (worldVelDir == Vector3.zero) ? Vector3.zero : (-worldVelDir * dragMagnitudeN);
                bodyLiftForce = Vector3.zero;
            }
        }

        // Helper function
        private static void CalculateDragCubeForces(object dragCubeListInstance, Part part, Vector3 localVelocityDir, float mach, float dynamicPressurePa,
                                            EnvironmentalData environment, GlobalAeroModifiers globals, Transform partTransform,
                                            out Vector3 dragForce, out Vector3 bodyLiftForce)
        {
            dragForce = Vector3.zero;
            bodyLiftForce = Vector3.zero;

            float[] areaOccludedList = (float[])_dclAreaOccludedField.GetValue(dragCubeListInstance);
            float[] weightedDragList = (float[])_dclWeightedDragField.GetValue(dragCubeListInstance);
            object surfaceCurvesListStruct = _dclSurfaceCurvesField.GetValue(dragCubeListInstance);
            object bodyLiftCurveStruct = _dclBodyLiftCurveField.GetValue(dragCubeListInstance); // This is PhysicsGlobals.LiftingSurfaceCurve for bodylift
            Vector3[] faceDirections = (Vector3[])_dclFaceDirectionsField.GetValue(null); // Static field
            float partBodyLiftMultiplier = (float)_partBodyLiftMultiplierField.GetValue(part);
            CustomLogger.Log("areaOccludedList", areaOccludedList);
            CustomLogger.Log("weightedDragList", weightedDragList);

            object dragCurveCd_DCL_Instance = _dclDragCurveCdField.GetValue(dragCubeListInstance);
            object dragCurveCdPower_DCL_Instance = _dclDragCurveCdPowerField.GetValue(dragCubeListInstance);

            double accumulatedAreaDragSum = 0;
            Vector3 accumulatedBodyLiftForceLocal = Vector3.zero;

            // UnityEngine.Debug.Log($"\n[SimulateAeroServices] Starting Drag Cube Face Loop");
            CustomLogger.Log("Starting drag cube face loop");
            CustomLogger.EnterFunction("FaceLoop");
            for (int i = 0; i < 6; i++)
            {
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Face Number {i}");
                CustomLogger.Log("Face num", i);
                Vector3 faceDir = faceDirections[i];
                float faceAngle = Vector3.Dot(localVelocityDir, faceDir);
                // UnityEngine.Debug.Log($"[SimulateAeroServices] faceDir: {faceDir}, faceDot: {faceAngle:F4}");
                CustomLogger.Log("faceDir", faceDir);
                CustomLogger.Log("faceAngle", faceAngle);
                // if (faceDot <= 0f) continue;

                float areaOccluded = areaOccludedList[i];
                // UnityEngine.Debug.Log($"[SimulateAeroServices] area: {areaOccluded:F4}");
                CustomLogger.Log("areaOccluded", areaOccluded);
                if (areaOccluded <= 0f) continue;

                float weightedDrag = weightedDragList[i];
                float final_cd_for_drag = weightedDrag;
                // UnityEngine.Debug.Log($"[SimulateAeroServices] original_cd: {weightedDrag:F4}");
                CustomLogger.Log("weightedDrag", weightedDrag);

                if (weightedDrag < 1.0f)
                {
                    float dragCurveCdEval = (float)_fcEvaluateMethod.Invoke(dragCurveCd_DCL_Instance, new object[] { weightedDrag });
                    float dragCurveCdPowerEval = (float)_fcEvaluateMethod.Invoke(dragCurveCdPower_DCL_Instance, new object[] { mach });
                    final_cd_for_drag = Mathf.Pow(dragCurveCdEval, dragCurveCdPowerEval);
                    // UnityEngine.Debug.Log($"[SimulateAeroServices] " +
                    //     $"dragCurveCdEval: {dragCurveCdEval:F4}, " +
                    //     $"dragCurveCdPowerEval: {dragCurveCdPowerEval:F4}, " +
                    //     $"final_cd_for_drag: {final_cd_for_drag:F4}");
                    CustomLogger.Log("dragCurveCdEval", dragCurveCdEval);
                    CustomLogger.Log("dragCurveCdPowerEval", dragCurveCdPowerEval);
                    CustomLogger.Log("final_cd_for_drag", final_cd_for_drag);
                }

                float dotNormalized = (faceAngle + 1f) * 0.5f;
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Entering EvaluateDragCurveValue()");
                CustomLogger.EnterFunction("EvalDragCurve");
                float dragCurveVal = EvaluateDragCurveValue(surfaceCurvesListStruct, dotNormalized, mach);
                CustomLogger.Log("dragCurveVal", dragCurveVal);
                CustomLogger.ExitFunction();
                // UnityEngine.Debug.Log($"[SimulateAeroServices] Exiting EvaluateDragCurveValue()");
                // UnityEngine.Debug.Log($"[SimulateAeroServices] dragCurveVal: {dragCurveVal:F4}");
                accumulatedAreaDragSum += areaOccluded * dragCurveVal * final_cd_for_drag;
                // UnityEngine.Debug.Log($"[SimulateAeroServices] accumulatedAreaDragSum: {accumulatedAreaDragSum:F4}");
                CustomLogger.Log("accumulatedAreaDragSum", accumulatedAreaDragSum);


                if (faceAngle > 0f) // <<<< THIS IS THE KEY CONDITION MATCHING KSP
                {
                    // UnityEngine.Debug.Log($"[SimulateAeroServices] faceDot > 0, calculating faceLift");
                    CustomLogger.Log("faceDot > 0, calculating faceLift");
                    if (bodyLiftCurveStruct != null) // Your reflection of PhysicsGlobals.BodyLiftCurve
                    {
                        FloatCurve bodyLiftCurve_FC = (FloatCurve)_lscLiftCurveField.GetValue(bodyLiftCurveStruct);
                        // Evaluate with faceDot, which is KSP's faceAngle. This is correct.
                        float bodyLiftCurveEval = (float)_fcEvaluateMethod.Invoke(bodyLiftCurve_FC, new object[] { faceAngle });
                        // UnityEngine.Debug.Log($"[SimulateAeroServices] liftCurveEval: {bodyLiftCurveEVal:F4}");
                        CustomLogger.Log("bodyLiftCurveEval", bodyLiftCurveEval);

                        if (!float.IsNaN(bodyLiftCurveEval) && bodyLiftCurveEval != 0f) // Good check, matches KSP's NaN check.
                        {
                            // Your per-face lift vector calculation:
                            Vector3 faceLift = -faceDir * (faceAngle * areaOccluded * weightedDrag * bodyLiftCurveEval);
                            // UnityEngine.Debug.Log($"[SimulateAeroServices] faceLift: {faceLift}");
                            CustomLogger.Log("faceLift", faceLift);
                            accumulatedBodyLiftForceLocal += faceLift;
                            // UnityEngine.Debug.Log($"[SimulateAeroServices] accumulatedBodyLiftForceLocal: {accumulatedBodyLiftForceLocal}");
                            CustomLogger.Log("accumulatedBodyLiftForceLocal", accumulatedBodyLiftForceLocal);
                        }
                    }
                }
            }
            CustomLogger.ExitFunction();

            double dragMagnitudeN = dynamicPressurePa * accumulatedAreaDragSum * globals.DragCubeMultiplier_PG * globals.PseudoReDragMult * globals.DragMultiplier;
            Vector3 dragVectorDir = (localVelocityDir == Vector3.zero) ? Vector3.zero : -partTransform.TransformDirection(localVelocityDir);
            dragForce = (dragVectorDir == Vector3.zero) ? Vector3.zero : (dragVectorDir * (float)dragMagnitudeN);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] dragMagnitudeN: {dragMagnitudeN}");
            CustomLogger.Log("dragMagnitudeN", dragMagnitudeN);

            if (bodyLiftCurveStruct != null && accumulatedBodyLiftForceLocal != Vector3.zero)
            {
                // UnityEngine.Debug.Log($"[SimulateAeroServices] bodyLift is non-zero, adding");
                CustomLogger.Log("bodyLift is non-zero, adding");
                FloatCurve bodyLiftMachCurve_FC = (FloatCurve)_lscLiftMachCurveField.GetValue(bodyLiftCurveStruct);
                float liftMachCurveEval = (float)_fcEvaluateMethod.Invoke(bodyLiftMachCurve_FC, new object[] { mach });
                float combinedBodyLiftMult = partBodyLiftMultiplier * globals.BodyLiftMultiplier * liftMachCurveEval;

                Vector3 worldBodyLiftComponent = accumulatedBodyLiftForceLocal * (float)dynamicPressurePa;
                worldBodyLiftComponent = partTransform.rotation * worldBodyLiftComponent * combinedBodyLiftMult;
                // UnityEngine.Debug.Log($"[SimulateAeroServices] worldBodyLiftComponent: {worldBodyLiftComponent}");
                CustomLogger.Log("worldBodyLiftComponent", worldBodyLiftComponent);

                if (dragVectorDir != Vector3.zero)
                {
                    bodyLiftForce = Vector3.ProjectOnPlane(worldBodyLiftComponent, dragVectorDir);
                    // UnityEngine.Debug.Log($"[SimulateAeroServices] bodyLiftForce: {bodyLiftForce}");
                    CustomLogger.Log("bodyLiftForce", bodyLiftForce);
                }
            }

        }

        // Helper function
        private static double CalculateSimpleDragValue(Part part, Part.DragModel dragModel, Vector3 localVelocityDir)
        {
            float maxDrag = (float)_partMaxDragField.GetValue(part);
            float minDrag = (float)_partMinDragField.GetValue(part);
            Vector3 refVector = (Vector3)_partDragReferenceVectorField.GetValue(part); // This is in local part space

            switch (dragModel)
            {
                case Part.DragModel.SPHERICAL:
                case Part.DragModel.DEFAULT:
                    return maxDrag;
                case Part.DragModel.CYLINDRICAL:
                    float dotCyl = Mathf.Abs(Vector3.Dot(refVector, localVelocityDir));
                    return Mathf.Lerp(minDrag, maxDrag, dotCyl);
                case Part.DragModel.CONIC:
                    float angleDeg = Vector3.Angle(refVector, localVelocityDir);
                    return Mathf.Lerp(minDrag, maxDrag, angleDeg / 180f);
                default:
                    return 0.0;
            }
        }

        // Helper function
        private static float EvaluateDragCurveValue(object surfaceCurvesListStructInstance, float dotNormalized, float mach)
        {
            Type surfaceCurvesListType = surfaceCurvesListStructInstance.GetType(); // Should be PhysicsGlobals.SurfaceCurvesList
            FieldInfo dragCurveTailField = surfaceCurvesListType.GetField("dragCurveTail", _flagsInstPub | _flagsInstNonPub);
            FieldInfo dragCurveSurfaceField = surfaceCurvesListType.GetField("dragCurveSurface", _flagsInstPub | _flagsInstNonPub);
            FieldInfo dragCurveMultiplierField = surfaceCurvesListType.GetField("dragCurveMultiplier", _flagsInstPub | _flagsInstNonPub);
            FieldInfo dragCurveTipField = surfaceCurvesListType.GetField("dragCurveTip", _flagsInstPub | _flagsInstNonPub);

            FloatCurve dragCurveTail = (FloatCurve)dragCurveTailField.GetValue(surfaceCurvesListStructInstance);
            FloatCurve dragCurveSurface = (FloatCurve)dragCurveSurfaceField.GetValue(surfaceCurvesListStructInstance);
            FloatCurve dragCurveMultiplierFc = (FloatCurve)dragCurveMultiplierField.GetValue(surfaceCurvesListStructInstance); // Renamed to avoid conflict
            FloatCurve dragCurveTip = (FloatCurve)dragCurveTipField.GetValue(surfaceCurvesListStructInstance);

            float result;
            if (dotNormalized <= 0.5f)
            {
                float tail = (float)_fcEvaluateMethod.Invoke(dragCurveTail, new object[] { mach });
                float surface = (float)_fcEvaluateMethod.Invoke(dragCurveSurface, new object[] { mach });
                result = Mathf.Lerp(tail, surface, dotNormalized * 2f);
            }
            else
            {
                float surface = (float)_fcEvaluateMethod.Invoke(dragCurveSurface, new object[] { mach });
                float tip = (float)_fcEvaluateMethod.Invoke(dragCurveTip, new object[] { mach });
                result = Mathf.Lerp(surface, tip, (dotNormalized - 0.5f) * 2f);
            }
            float evaluatedMultiplier = (float)_fcEvaluateMethod.Invoke(dragCurveMultiplierFc, new object[] { mach });
            return result * evaluatedMultiplier;
        }

        // Helper function
        private static void CalculateModuleLiftingSurfaceForce(
            ModuleLiftingSurface mls, Part part, Vector3 worldVelDir, float speed, float mach,
            float dynamicPressurePa, EnvironmentalData environment, GlobalAeroModifiers globals,
            out Vector3 liftForce, out Vector3 dragForce)
        {
            liftForce = Vector3.zero;
            dragForce = Vector3.zero;

            Transform baseTransform = (Transform)_mlsBaseTransformField.GetValue(mls);
            if (baseTransform == null) baseTransform = (Transform)_partTransformField.GetValue(part);

            Vector3 nVel, liftVector;
            float liftDot, absDot;
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Entering SetupMLSCoefficients()");
            CustomLogger.EnterFunction("SetupMLS");
            SetupMLSCoefficients(mls, baseTransform, worldVelDir, out nVel, out liftVector, out liftDot, out absDot);
            CustomLogger.Log("nVel", nVel);
            CustomLogger.Log("liftVector", liftVector);
            CustomLogger.Log("liftDot", liftDot);
            CustomLogger.Log("absDot", absDot);
            CustomLogger.ExitFunction();
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Exiting SetupMLSCoefficients()");
            // UnityEngine.Debug.Log($"[SimulateAeroServices] nVel_module: {nVel}, liftAxisN_module: {liftVector}, liftDot_module: {liftDot:F4}, absDot_module: {absDot:F4}");

            // UnityEngine.Debug.Log($"[SimulateAeroServices] Entering CalculateSingleSurfaceAero()");
            CustomLogger.EnterFunction("CalcSingleSurface");
            CalculateSingleSurfaceAero(mls, baseTransform, worldVelDir, speed, mach, dynamicPressurePa, environment, globals,
                                       nVel, liftVector, liftDot, absDot, part,
                                       out liftForce, out dragForce);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] Exiting CalculateSingleSurfaceAero()");
            CustomLogger.ExitFunction();
        }

        // Helper function
        private static void CalculateModuleControlSurfaceForce(
            ModuleControlSurface mcs, Part part, Vector3 worldVelDir, float speed, float mach,
            float dynamicPressurePa, EnvironmentalData environment, GlobalAeroModifiers globals,
            SimAeroFlightControlStateInput controlInputs, Vector3 vesselCurrentCoM, Transform vesselReferenceTransform,
            out Vector3 ctrlSurfLiftForce, out Vector3 ctrlSurfDragForce)
        {
            ctrlSurfLiftForce = Vector3.zero;
            ctrlSurfDragForce = Vector3.zero;

            Transform baseTransform = (Transform)_mlsBaseTransformField.GetValue(mcs); // Inherited
            if (baseTransform == null) baseTransform = (Transform)_partTransformField.GetValue(part);

            // Calculate Deflection
            float pitchInput = (bool)_mcsIgnorePitchField.GetValue(mcs) ? 0f : controlInputs.Pitch;
            float yawInput = (bool)_mcsIgnoreYawField.GetValue(mcs) ? 0f : controlInputs.Yaw;
            float rollInputVal = (bool)_mcsIgnoreRollField.GetValue(mcs) ? 0f : controlInputs.Roll;

            Vector3 localCoM = baseTransform.InverseTransformPoint(vesselCurrentCoM);
            Vector3 worldCtrlInputFromPitchYaw = vesselReferenceTransform.rotation * new Vector3(pitchInput, 0f, yawInput);
            float dynamicAction = Vector3.Dot(worldCtrlInputFromPitchYaw, baseTransform.right);
            if (localCoM.y < 0f) dynamicAction = -dynamicAction;

            float rollComponent = 0f;
            if (!(bool)_mcsIgnoreRollField.GetValue(mcs))
            {
                Vector3 localCoM_XZ_Norm = new Vector3(localCoM.x, 0f, localCoM.z).normalized;
                rollComponent = Vector3.Dot(Vector3.right, localCoM_XZ_Norm) *
                               (1f - (Mathf.Abs(Vector3.Dot(localCoM_XZ_Norm, Quaternion.Inverse(baseTransform.rotation) * vesselReferenceTransform.up)) * 0.5f + 0.5f)) *
                               Mathf.Sign(Vector3.Dot(baseTransform.up, vesselReferenceTransform.up)) *
                               rollInputVal *
                               Mathf.Sign((float)_mcsCtrlSurfaceRangeField.GetValue(mcs));
            }
            dynamicAction = Mathf.Clamp(dynamicAction - rollComponent, -1f, 1f);

            float ctrlSurfaceRange = (float)_mcsCtrlSurfaceRangeField.GetValue(mcs);
            float authorityLimiter = (float)_mcsAuthorityLimiterField.GetValue(mcs);
            float deflectionDirection = (float)_mcsDeflectionDirectionField.GetValue(mcs);

            float totalDeflectionInput = dynamicAction * ctrlSurfaceRange * (authorityLimiter / 100f);

            if ((bool)_mcsDeployField.GetValue(mcs))
            {
                bool deployInvert = (bool)_mcsDeployInvertField.GetValue(mcs);
                float deploySign = deployInvert ? -1f : 1f;
                // Simplified: KSP has usesMirrorDeploy and partDeployInvert for editor symmetry logic, ignoring for sim.
                float currentDeployAngle = deploySign * (float)_mcsDeployAngleField.GetValue(mcs); // * deflectionDirection applied later to total
                totalDeflectionInput += currentDeployAngle;
            }

            float deflection = Mathf.Clamp(totalDeflectionInput * deflectionDirection, -1.5f * ctrlSurfaceRange, 1.5f * ctrlSurfaceRange);
            // Note: Actual KSP ModuleControlSurface moves deflection towards target by actuatorSpeed over time. We assume instant.

            // --- Aerodynamic Calculation ---
            Vector3 nVel_undeflected, liftAxisN_undeflected;
            float liftDot_undeflected, absDot_undeflected;
            CustomLogger.EnterFunction("SetupMLS");
            SetupMLSCoefficients(mcs, baseTransform, worldVelDir, out nVel_undeflected, out liftAxisN_undeflected, out liftDot_undeflected, out absDot_undeflected);
            CustomLogger.Log("nVel", nVel_undeflected);
            CustomLogger.Log("liftVector", liftAxisN_undeflected);
            CustomLogger.Log("liftDot", liftDot_undeflected);
            CustomLogger.Log("absDot", absDot_undeflected);
            CustomLogger.ExitFunction();

            float ctrlSurfaceAreaFraction = (float)_mcsCtrlSurfaceAreaField.GetValue(mcs);
            float fixedAreaFraction = 1.0f - ctrlSurfaceAreaFraction;

            Vector3 fixedLift, fixedDrag;
            if (fixedAreaFraction > 1e-6f) // Calculate if fixed part exists
            {
                CustomLogger.EnterFunction("CalcSingleSurface");
                CalculateSingleSurfaceAero(mcs, baseTransform, worldVelDir, speed, mach, dynamicPressurePa, environment, globals,
                                           nVel_undeflected, liftAxisN_undeflected, liftDot_undeflected, absDot_undeflected, part,
                                           out fixedLift, out fixedDrag);
                CustomLogger.ExitFunction();
                ctrlSurfLiftForce += fixedLift * fixedAreaFraction;
                ctrlSurfDragForce += fixedDrag * fixedAreaFraction;
            }

            if (ctrlSurfaceAreaFraction > 1e-6f) // Calculate if control surface part exists
            {
                Quaternion airflowIncidence = Quaternion.AngleAxis(-deflection, baseTransform.right); // Deflection is around local X
                Vector3 liftAxisN_deflected = airflowIncidence * liftAxisN_undeflected;

                float liftDot_deflected = Vector3.Dot(nVel_undeflected, liftAxisN_deflected);
                bool omni = (bool)_mlsOmnidirectionalField.GetValue(mcs);
                float absDot_deflected = omni ? Mathf.Abs(liftDot_deflected) : Mathf.Clamp01(liftDot_deflected);

                Vector3 deflectedLift, deflectedDrag;
                CustomLogger.EnterFunction("CalcSingleSurface");
                CalculateSingleSurfaceAero(mcs, baseTransform, worldVelDir, speed, mach, dynamicPressurePa, environment, globals,
                                           nVel_undeflected, liftAxisN_deflected, liftDot_deflected, absDot_deflected, part,
                                           out deflectedLift, out deflectedDrag);
                CustomLogger.ExitFunction();
                ctrlSurfLiftForce += deflectedLift * ctrlSurfaceAreaFraction;
                ctrlSurfDragForce += deflectedDrag * ctrlSurfaceAreaFraction;
            }
        }

        // Helper function
        private static void SetupMLSCoefficients(
            ModuleLiftingSurface mls, Transform baseTransformToUse, Vector3 worldVelDir,
            out Vector3 nVel, out Vector3 liftVector,
            out float liftDot, out float absDot)
        {
            nVel = worldVelDir;
            ModuleLiftingSurface.TransformDir transformDir = (ModuleLiftingSurface.TransformDir)_mlsTransformDirField.GetValue(mls);
            float transformSign = (float)_mlsTransformSignField.GetValue(mls);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] transformSign: {transformSign:F4}");
            CustomLogger.Log("transformSign", transformSign);

            switch (transformDir)
            {
                case ModuleLiftingSurface.TransformDir.Z: liftVector = baseTransformToUse.forward; CustomLogger.Log("Z TransformDir"); break;
                case ModuleLiftingSurface.TransformDir.Y: liftVector = baseTransformToUse.up; CustomLogger.Log("Y TransformDir"); break;
                case ModuleLiftingSurface.TransformDir.X: default: liftVector = baseTransformToUse.right; CustomLogger.Log("X TransformDir"); break;
            }
            liftVector *= transformSign;

            liftDot = Vector3.Dot(nVel, liftVector);
            bool omnidirectional = (bool)_mlsOmnidirectionalField.GetValue(mls);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] omnidirectional: {omnidirectional}");
            CustomLogger.Log("omnidirectional", omnidirectional);
            absDot = omnidirectional ? Mathf.Abs(liftDot) : Mathf.Clamp01(liftDot);
        }

        // Helper function
        private static void CalculateSingleSurfaceAero(
            ModuleLiftingSurface mlsInstance, Transform mlsBaseTransform, Vector3 worldVelDir, float speed, float mach,
            float dynamicPressurePa, EnvironmentalData environment, GlobalAeroModifiers globals,
            Vector3 nVel, Vector3 liftVector, float liftDot, float absDot, Part part,
            out Vector3 liftForceResult, out Vector3 dragForceResult)
        {
            liftForceResult = Vector3.zero;
            dragForceResult = Vector3.zero;

            // Get nodeEnabled value
            bool nodeEnabledValue = (bool)_mlsNodeEnabledField.GetValue(mlsInstance);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] mlsInstance: {mlsInstance}, nodeEnabledValue: {nodeEnabledValue}");
            CustomLogger.Log("mlsInstance", mlsInstance);
            CustomLogger.Log("nodeEnabled", nodeEnabledValue);

            if (nodeEnabledValue)
            {
                // Option 1: Directly access the 'attachNode' field from ModuleLiftingSurface
                object attachNodeObject = _mlsAttachNodeField.GetValue(mlsInstance);

                // Option 2: Replicate the logic from ModuleLiftingSurface.OnStart()
                // This is more robust if 'attachNode' might not be initialized in your context.
                if (attachNodeObject == null) // If direct access yielded null, try to find it
                {
                    string attachNodeNameString = (string)_mlsAttachNodeNameField.GetValue(mlsInstance);
                    if (!string.IsNullOrEmpty(attachNodeNameString))
                    {
                        // Call Part.FindAttachNode(string nodeName)
                        // Ensure 'part' is the correct Part instance associated with mlsInstance
                        attachNodeObject = _partFindAttachNodeMethod.Invoke(part, new object[] { attachNodeNameString });
                        // UnityEngine.Debug.Log($"[SimulateAeroServices] Looked up attachNode '{attachNodeNameString}': {(attachNodeObject != null)}");
                        CustomLogger.Log("Looked up attachNode", attachNodeNameString, (attachNodeObject != null));
                    }
                }

                if (attachNodeObject != null)
                {
                    // Get attachedPart from the AttachNode object
                    object attachedPartObject = _anAttachedPartField.GetValue(attachNodeObject);
                    // UnityEngine.Debug.Log($"[SimulateAeroServices] attachNodeObject: {attachNodeObject}, attachedPartObject: {attachedPartObject}");
                    CustomLogger.Log("attachNodeObject", attachNodeObject);
                    CustomLogger.Log("attachedPartObject", attachedPartObject);

                    if (attachedPartObject != null) // This means a part is attached
                    {
                        // UnityEngine.Debug.Log($"[SimulateAeroServices] Node enabled and part attached. Zeroing out lift/drag for this surface.");
                        CustomLogger.Log("Node enabled and part attached. Zeroing out lift/drag for this surface");
                        // Original KSP code sets liftScalar = 0 and returns Vector3.zero.
                        // We do the same by ensuring liftForceResult and dragForceResult remain zero and returning.
                        return; // Exit early, no lift or drag from this surface
                    }
                }
            }

            float deflectionLiftCoeff = (float)_mlsDeflectionLiftCoeffField.GetValue(mlsInstance);
            bool perpendicularOnly = (bool)_mlsPerpendicularOnlyField.GetValue(mlsInstance);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] deflectionLiftCoeff: {deflectionLiftCoeff:F4}, perpendicularOnly: {perpendicularOnly}");
            CustomLogger.Log("deflectionLiftCoeff", deflectionLiftCoeff);
            CustomLogger.Log("perpendicularOnly", perpendicularOnly);
            FloatCurve liftCurve = (FloatCurve)_mlsLiftCurveField.GetValue(mlsInstance);
            FloatCurve liftMachCurve = (FloatCurve)_mlsLiftMachCurveField.GetValue(mlsInstance);

            float liftCurveEval = (float)_fcEvaluateMethod.Invoke(liftCurve, new object[] { absDot });
            float liftMachCurveEval = (float)_fcEvaluateMethod.Invoke(liftMachCurve, new object[] { mach });
            float liftScalar = Mathf.Sign(liftDot) * liftCurveEval * liftMachCurveEval * deflectionLiftCoeff * globals.LiftMultiplier * dynamicPressurePa;
            // UnityEngine.Debug.Log($"[SimulateAeroServices] liftCurveEval: {liftCurveEval:F4}, liftMachCurveEval: {liftMachCurveEval:F4}, liftScalarN: {liftScalar:F4}");
            CustomLogger.Log("liftCurveEval", liftCurveEval);
            CustomLogger.Log("liftMachCurveEval", liftMachCurveEval);
            CustomLogger.Log("liftScalar", liftScalar);

            if (liftScalar != 0f && !float.IsNaN(liftScalar))
            {
                liftForceResult = -liftVector * liftScalar;
                if (perpendicularOnly)
                {
                    liftForceResult = Vector3.ProjectOnPlane(liftForceResult, -nVel);
                }
            }

            bool useInternalDrag = (bool)_mlsUseInternalDragModelField.GetValue(mlsInstance);
            // UnityEngine.Debug.Log($"[SimulateAeroServices] useInternalDrag: {useInternalDrag}");
            CustomLogger.Log("useInternalDrag", useInternalDrag);
            if (useInternalDrag)
            {
                FloatCurve dragCurve = (FloatCurve)_mlsDragCurveField.GetValue(mlsInstance);
                FloatCurve dragMachCurve = (FloatCurve)_mlsDragMachCurveField.GetValue(mlsInstance);

                float dragCurveEval = (float)_fcEvaluateMethod.Invoke(dragCurve, new object[] { absDot });
                float dragMachCurveEval = (float)_fcEvaluateMethod.Invoke(dragMachCurve, new object[] { mach });
                float dragScalar = dragCurveEval * dragMachCurveEval * deflectionLiftCoeff * globals.LiftDragMultiplier * dynamicPressurePa;
                // UnityEngine.Debug.Log($"[SimulateAeroServices] dragCurveEval: {dragCurveEval:F4}, dragMachCurveEval: {dragMachCurveEval:F4}, dragScalarN: {dragScalar:F4}");
                CustomLogger.Log("dragCurveEval", dragCurveEval);
                CustomLogger.Log("dragMachCurveEval", dragMachCurveEval);
                CustomLogger.Log("dragScalar", dragScalar);

                if (dragScalar != 0f && !float.IsNaN(dragScalar) && worldVelDir != Vector3.zero)
                {
                    dragForceResult = -nVel * dragScalar; // Drag is opposite to velocity vector
                }
            }
        }

    } // End class SimulateAerodynamicsService
} // End namespace