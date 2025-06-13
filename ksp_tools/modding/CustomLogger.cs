using System;
using System.Collections.Generic;
using System.Globalization; // Required for number formatting with CultureInfo.InvariantCulture
using System.Linq;
using System.Text;
// Assuming you'll use this in a Unity environment based on your original code.
// If not, you can replace UnityEngine.Debug.Log with System.Console.WriteLine or another logging framework.
using UnityEngine;

namespace CustomLogging
{
    public static class CustomLogger
    {
        private static readonly Stack<string> _functionContextStack = new Stack<string>();
        private static string _rootContext = "DefaultApp"; // Default root context
        private static int _numericPrecision = 4; // Default numeric precision
        private static bool _isVerbose = true; // Default to verbose logging

        /// <summary>
        /// Gets or sets the root context string for the logger.
        /// Changing this will reset the current context stack to the new root.
        /// </summary>
        public static string RootContext
        {
            get => _rootContext;
            set
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    _rootContext = "DefaultApp"; // Fallback to a default if null/empty
                    if (_isVerbose) UnityEngine.Debug.LogWarning("[CustomLogger.RootContext] Root context cannot be null or empty. Using 'DefaultApp'.");
                }
                else
                {
                    _rootContext = value;
                }
                // When root context changes, reset the stack to reflect the new root.
                ForceResetContext();
            }
        }

        /// <summary>
        /// Gets or sets the number of decimal places for formatting floating-point numbers.
        /// Default is 4.
        /// </summary>
        public static int NumericPrecision
        {
            get => _numericPrecision;
            set
            {
                if (value < 0)
                {
                    _numericPrecision = 0; // Precision cannot be negative
                    if (_isVerbose) UnityEngine.Debug.LogWarning("[CustomLogger.NumericPrecision] Numeric precision cannot be negative. Setting to 0.");
                }
                else
                {
                    _numericPrecision = value;
                }
            }
        }

        /// <summary>
        /// Gets or sets whether verbose logging (Log, LogWarning) is enabled.
        /// Error logs (LogError) are always enabled.
        /// Defaults to true (verbose logging enabled).
        /// </summary>
        public static bool IsVerbose
        {
            get => _isVerbose;
            set => _isVerbose = value;
        }


        static CustomLogger()
        {
            // Initialize with the initial root context
            ResetContext();
        }

        private static void ResetContext()
        {
            _functionContextStack.Clear();
            _functionContextStack.Push(_rootContext); // Use the potentially configured RootContext
        }

        /// <summary>
        /// Call this when entering a function to add it to the context stack.
        /// </summary>
        /// <param name="functionName">The name of the function being entered.</param>
        public static void EnterFunction(string functionName)
        {
            if (string.IsNullOrWhiteSpace(functionName))
            {
                _functionContextStack.Push("UnnamedFunction");
                if (_isVerbose) UnityEngine.Debug.LogWarning("[CustomLogger.EnterFunction] Function name was null or empty. Pushed 'UnnamedFunction'.");
                return;
            }
            _functionContextStack.Push(functionName);
        }

        /// <summary>
        /// Call this when exiting a function to remove it from the context stack.
        /// Ensures the root context is not removed.
        /// </summary>
        public static void ExitFunction()
        {
            if (_functionContextStack.Count > 1) // Keep the root context
            {
                _functionContextStack.Pop();
            }
            else
            {
                // This warning itself should be subject to IsVerbose, or be an LogError if critical
                if (_isVerbose) UnityEngine.Debug.LogWarning("[CustomLogger.ExitFunction] Attempted to pop the root context. Stack remains at root.");
            }
        }

        /// <summary>
        /// Resets the function context stack to its initial state (e.g., ['ConfiguredRootContext']).
        /// </summary>
        public static void ForceResetContext()
        {
            ResetContext();
        }

        private static string GetContextPrefix()
        {
            if (_functionContextStack.Count == 0)
            {
                 ResetContext();
            }
            return $"[{string.Join(".", _functionContextStack.Reverse())}] ";
        }

        private static string FormatData(object data)
        {
            if (data == null) return "null";

            string formatString = $"F{_numericPrecision}";

            if (data is float f) return f.ToString(formatString, CultureInfo.InvariantCulture);
            if (data is double d) return d.ToString(formatString, CultureInfo.InvariantCulture);
            if (data is decimal m) return m.ToString(formatString, CultureInfo.InvariantCulture);
            if (data is UnityEngine.Vector2 v2) return $"({v2.x.ToString(formatString, CultureInfo.InvariantCulture)}, {v2.y.ToString(formatString, CultureInfo.InvariantCulture)})";
            if (data is UnityEngine.Vector3 v3) return $"({v3.x.ToString(formatString, CultureInfo.InvariantCulture)}, {v3.y.ToString(formatString, CultureInfo.InvariantCulture)}, {v3.z.ToString(formatString, CultureInfo.InvariantCulture)})";
            if (data is UnityEngine.Quaternion q) return $"({q.x.ToString(formatString, CultureInfo.InvariantCulture)}, {q.y.ToString(formatString, CultureInfo.InvariantCulture)}, {q.z.ToString(formatString, CultureInfo.InvariantCulture)}, {q.w.ToString(formatString, CultureInfo.InvariantCulture)})";

            return data.ToString();
        }

        /// <summary>
        /// Logs a message with the current function context if IsVerbose is true.
        /// </summary>
        /// <param name="message">The message to log.</param>
        public static void Log(string message)
        {
            if (!_isVerbose) return;
            UnityEngine.Debug.Log(GetContextPrefix() + message);
        }

        /// <summary>
        /// Logs a message and a single data object with the current function context if IsVerbose is true.
        /// Applies numeric formatting if applicable.
        /// </summary>
        /// <param name="message">The message to log.</param>
        /// <param name="data">The data object to log.</param>
        public static void Log(string message, object data)
        {
            if (!_isVerbose) return;
            StringBuilder sb = new StringBuilder();
            sb.Append(GetContextPrefix());
            sb.Append(message);
            // sb.Append(" Data: ");
            sb.Append(": ")
            sb.Append(FormatData(data));
            UnityEngine.Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Logs a message and multiple data objects with the current function context if IsVerbose is true.
        /// Applies numeric formatting to each data object if applicable.
        /// </summary>
        /// <param name="message">The message to log.</param>
        /// <param name="dataArgs">The data objects to log.</param>
        public static void Log(string message, params object[] dataArgs)
        {
            if (!_isVerbose) return;
            StringBuilder sb = new StringBuilder();
            sb.Append(GetContextPrefix());
            sb.Append(message);

            if (dataArgs != null && dataArgs.Length > 0)
            {
                // sb.Append(" Data: [");
                sb.Append(": [")
                for (int i = 0; i < dataArgs.Length; i++)
                {
                    sb.Append(FormatData(dataArgs[i]));
                    if (i < dataArgs.Length - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.Append("]");
            }
            UnityEngine.Debug.Log(sb.ToString());
        }

        /// <summary>
        /// Logs an error message with the current function context. This log is NOT affected by IsVerbose.
        /// </summary>
        /// <param name="message">The error message to log.</param>
        public static void LogError(string message)
        {
            UnityEngine.Debug.LogError(GetContextPrefix() + message);
        }
        
        /// <summary>
        /// Logs an error message and an exception with the current function context. This log is NOT affected by IsVerbose.
        /// </summary>
        /// <param name="message">The error message to log.</param>
        /// <param name="exception">The exception to log.</param>
        public static void LogError(string message, Exception exception)
        {
            UnityEngine.Debug.LogError(GetContextPrefix() + message + "\nException: " + exception?.ToString());
        }

        /// <summary>
        /// Logs a warning message with the current function context if IsVerbose is true.
        /// </summary>
        /// <param name="message">The warning message to log.</param>
        public static void LogWarning(string message)
        {
            if (!_isVerbose) return;
            UnityEngine.Debug.LogWarning(GetContextPrefix() + message);
        }
    }

    // // Example of how to use the CustomLogger
    // public class ExampleUsageVerbose
    // {
    //     public static void RunExample()
    //     {
    //         CustomLogger.RootContext = "VerboseDemo";
    //         CustomLogger.NumericPrecision = 2;

    //         CustomLogger.Log("--- VERBOSE LOGGING ON (DEFAULT) ---");
    //         CustomLogger.IsVerbose = true; // Explicitly set, though it's default

    //         CustomLogger.EnterFunction("TestFunctionVerbose");
    //         CustomLogger.Log("This is a regular log message.");
    //         CustomLogger.Log("Some data", 123.456f, true);
    //         CustomLogger.LogWarning("This is a warning message.");
    //         CustomLogger.LogError("This is an error message, always shown.");
    //         CustomLogger.ExitFunction();

    //         CustomLogger.Log("--- VERBOSE LOGGING OFF ---");
    //         CustomLogger.IsVerbose = false;

    //         CustomLogger.EnterFunction("TestFunctionNonVerbose"); // Context stack is still managed
    //         CustomLogger.Log("This regular log WILL NOT BE SHOWN.");
    //         CustomLogger.Log("This data WILL NOT BE SHOWN.", 789.012f, false);
    //         CustomLogger.LogWarning("This warning WILL NOT BE SHOWN.");
    //         CustomLogger.LogError("This is another error message, ALWAYS SHOWN even when not verbose.");
    //         CustomLogger.ExitFunction();


    //         CustomLogger.Log("--- VERBOSE LOGGING RE-ENABLED ---");
    //         CustomLogger.IsVerbose = true;
    //         CustomLogger.EnterFunction("FinalCheck");
    //         CustomLogger.Log("Verbose logging is back on.");
    //         CustomLogger.ExitFunction();

    //         CustomLogger.Log("Example finished.");
    //     }
    // }
}

// To test this (e.g., in a Unity script's Start() method or a console app's Main()):
// CustomLogging.ExampleUsageVerbose.RunExample();

/*
Expected output from ExampleUsageVerbose.RunExample():

[VerboseDemo] --- VERBOSE LOGGING ON (DEFAULT) ---
[VerboseDemo.TestFunctionVerbose] This is a regular log message.
[VerboseDemo.TestFunctionVerbose] Some data Data: [123.46, True]
[VerboseDemo.TestFunctionVerbose] This is a warning message.
[VerboseDemo.TestFunctionVerbose] This is an error message, always shown.
[VerboseDemo] --- VERBOSE LOGGING OFF ---
[VerboseDemo.TestFunctionNonVerbose] This is another error message, ALWAYS SHOWN even when not verbose.
[VerboseDemo] --- VERBOSE LOGGING RE-ENABLED ---
[VerboseDemo.FinalCheck] Verbose logging is back on.
[VerboseDemo] Example finished.
*/