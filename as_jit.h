#include "angelscript.h"
#include <vector>
#include <map>

namespace assembler {
struct CodePage;
};

enum JITSettings {
	//Should the JIT attempt to suspend? (Slightly faster, but makes suspension very rare if it occurs at all)
	JIT_NO_SUSPEND = 0x01,
	//Should the JIT reset the FPU entering System calls? (Slightly faster, may not work on all platforms)
	JIT_SYSCALL_FPU_NORESET = 0x02,
	//Should the JIT support error events from System calls? (Faster, but exceptions will generally be ignored, possibly leading to crashes)
	JIT_SYSCALL_NO_ERRORS = 0x04,
	//Do allocation/deallocation functions inspect the script context? (Faster, but won't work correctly if you try to get information about the script system during allocations)
	JIT_ALLOC_SIMPLE = 0x08,
	//Fall back to AngelScript to perform switch logic? (Slower, but uses less memory)
	JIT_NO_SWITCHES = 0x10,
};

class asCJITCompiler : public asIJITCompiler {
	unsigned flags;
	assembler::CodePage* activePage;
	std::multimap<asJITFunction,assembler::CodePage*> pages;
	std::map<asJITFunction,unsigned char**> jumpTables;
public:
	asCJITCompiler(unsigned Flags = 0);
	int CompileFunction(asIScriptFunction *function, asJITFunction *output);
    void ReleaseJITFunction(asJITFunction func);
	void finalizePages();
};
