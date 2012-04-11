#include "as_jit.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <map>
#include <functional>

#include "../source/as_scriptfunction.h"
#include "../source/as_objecttype.h"
#include "../source/as_callfunc.h"
#include "../source/as_scriptengine.h"
#include "../source/as_scriptobject.h"
#include "../source/as_texts.h"

#include "virtual_asm.h"
using namespace assembler;

CriticalSection jitLock;

#ifdef __amd64__
#define stdcall
#define JIT_64
#endif

#ifdef JIT_64
#define stdcall
#else
#ifdef _MSC_VER
#define stdcall __stdcall
#else
#define stdcall __attribute__((stdcall))
#endif
#endif

const unsigned codePageSize = 10000;

#define offset0 (asBC_SWORDARG0(pOp)*sizeof(asDWORD))
#define offset1 (asBC_SWORDARG1(pOp)*sizeof(asDWORD))
#define offset2 (asBC_SWORDARG2(pOp)*sizeof(asDWORD))

short offset(asDWORD* op, unsigned n) {
	return *(((short*)op) + (n+1)) * sizeof(asDWORD);
}

//Wrappers so we can deal with complex pointers/calling conventions
void stdcall popStackIfNotEmpty(asIScriptContext* ctx);

void stdcall allocScriptObject(asCObjectType* type, asCScriptFunction* constructor, asIScriptEngine* engine, asSVMRegisters* registers);

void* stdcall engineAlloc(asCScriptEngine* engine, asCObjectType* type);

void stdcall engineRelease(asCScriptEngine* engine, void* memory, asCScriptFunction* release);

void stdcall engineDestroyFree(asCScriptEngine* engine, void* memory, asCScriptFunction* destruct);

void stdcall engineFree(asCScriptEngine* engine, void* memory);

void stdcall engineCallMethod(asCScriptEngine* engine, void* object, asCScriptFunction* method);

void stdcall callScriptFunction(asIScriptContext* ctx, asCScriptFunction* func);

void stdcall callInterfaceMethod(asIScriptContext* ctx, asCScriptFunction* func);

size_t stdcall callBoundFunction(asIScriptContext* ctx, unsigned short fid);

asCScriptObject* stdcall castObject(asCScriptObject* obj, asCObjectType* to);

bool stdcall doSuspend(asIScriptContext* ctx);

//Wrapper functions to cast between types, or perform math on large types, where doing so is overly complicated in the ASM
#ifdef _MSC_VER
template<class F, class T>
void stdcall directConvert(F* from, T* to) {
	*to = (T)*from;
}
#else
//stdcall doesn't work with templates on GCC
template<class F, class T>
void directConvert(F* from, T* to) {
	*to = (T)*from;
}
#endif

float stdcall fmod_wrapper_f(float* div, float* mod) {
	return fmod(*div, *mod);
}

double stdcall fmod_wrapper(double* div, double* mod) {
	return fmod(*div, *mod);
}

void stdcall i64_add(long long* a, long long* b, long long* r) {
	*r = *a + *b;
}

void stdcall i64_sub(long long* a, long long* b, long long* r) {
	*r = *a - *b;
}

void stdcall i64_mul(long long* a, long long* b, long long* r) {
	*r = *a * *b;
}

void stdcall i64_div(long long* a, long long* b, long long* r) {
	*r = *a / *b;
}

void stdcall i64_mod(long long* a, long long* b, long long* r) {
	*r = *a % *b;
}

void stdcall i64_sll(unsigned long long* a, asDWORD* b, unsigned long long* r) {
	*r = *a << *b;
}

void stdcall i64_srl(unsigned long long* a, asDWORD* b, unsigned long long* r) {
	*r = *a << *b;
}

void stdcall i64_sra(long long* a, asDWORD* b, long long* r) {
	*r = *a << *b;
}

int stdcall cmp_int64(long long* a, long long* b) {
	long long c = *a - *b;
	if( c == 0)
		return 0;
	else if( c < 0)
		return -1;
	else
		return 1;
}

int stdcall cmp_uint64(unsigned long long* a, unsigned long long* b) {
	unsigned long long c = *a - *b;
	if( c == 0)
		return 0;
	else if( c < 0)
		return -1;
	else
		return 1;
}

size_t stdcall div_ull(unsigned long long* div, unsigned long long* by, unsigned long long* result) {
	if(*by == 0)
		return 1;
	*result = *div / *by;
	return 0;
}

size_t stdcall mod_ull(unsigned long long* div, unsigned long long* by, unsigned long long* result) {
	if(*by == 0)
		return 1;
	*result = *div % *by;
	return 0;
}

enum ObjectPosition {
	OP_This,
	OP_First,
	OP_Last,
	OP_None
};

struct SystemCall {
	Processor& cpu;
	FloatingPointUnit& fpu;
	asDWORD* const & pOp;
	unsigned flags;
	std::function<void(JumpType)> returnHandler;

	SystemCall(Processor& CPU, FloatingPointUnit& FPU,
		std::function<void(JumpType)> ConditionalReturn, asDWORD* const & bytecode, unsigned Flags)
		: cpu(CPU), fpu(FPU), returnHandler(ConditionalReturn), pOp(bytecode), flags(Flags) {}

	void callSystemFunction(asCScriptFunction* func, Register* objPointer = 0);

private:
	void call_viaAS(asCScriptFunction* func, Register* objPointer);
	void call_stdcall(asSSystemFunctionInterface* func, asCScriptFunction* sFunc);
	void call_cdecl(asSSystemFunctionInterface* func, asCScriptFunction* sFunc);
	void call_cdecl_obj(asSSystemFunctionInterface* func, asCScriptFunction* sFunc, Register* objPointer, bool last);
	void call_thiscall(asSSystemFunctionInterface* func, asCScriptFunction* sFunc, Register* objPointer);

	void call_64conv(asSSystemFunctionInterface* func, asCScriptFunction* sFunc, Register* objPointer, ObjectPosition pos);

	void call_getPrimitiveReturn(asSSystemFunctionInterface* func);
	
	//Handles error handling
	void call_entry(asSSystemFunctionInterface* func, asCScriptFunction* sFunc);
	void call_error();
	void call_exit(asSSystemFunctionInterface* func);
};

unsigned toSize(asEBCInstr instr) {
	return asBCTypeSize[asBCInfo[instr].type];
}

asCJITCompiler::asCJITCompiler(unsigned Flags) : flags(Flags), activePage(0) {
}

//Returns the total number of bytes that will be pushed, until the next op that doesn't push
unsigned findTotalPushBatchSize(asDWORD* firstPush, asDWORD* endOfBytecode);

int asCJITCompiler::CompileFunction(asIScriptFunction *function, asJITFunction *output) {
	asUINT   length;
	asDWORD *pOp = function->GetByteCode(&length);

	//No bytecode for this function, don't bother making any jit for it
	if(pOp == 0 || length == 0) {
		output = 0;
		return 1;
	}

	asDWORD *end = pOp + length, *start = pOp;

	volatile byte** jumpTable = new volatile byte*[length];
	memset(jumpTable,0,length * sizeof(void*));
	bool tableInUse = false;

	std::multimap<asDWORD*, void*> futureJumps;

	jitLock.enter();

	//Get the active page, or create a new one if the current one is missing or too small (256 bytes for the entry and a few ops)
	if(activePage == 0 || activePage->final || activePage->getFreeSize() < 256)
		activePage = new CodePage(codePageSize);
	else
		activePage->grab();

	*output = activePage->getFunctionPointer<asJITFunction>();
	pages.insert(std::pair<asJITFunction,assembler::CodePage*>(*output,activePage));

	//If we are outside of opcodes we can execute, ignore all ops until a new JIT entry is found
	bool waitingForEntry = true;

	//Setup the processor as a 32 bit processor, as most angelscript ops work on integers
	Processor cpu(*activePage, 32);
	byte* byteStart = (byte*)cpu.op;

	FloatingPointUnit fpu(cpu);

	unsigned pBits = sizeof(void*) * 8;

#ifdef JIT_64
	//64-bit uses registers for function arguments, so use some other
	//non-volatile registers for our arguments

	//32-bit registers
	Register eax(cpu,EAX), ecx(cpu,EBX), edx(cpu,EDX), ebp(cpu,R15), edi(cpu,R13);
	Register rarg(cpu, R10);
	//8-bit registers
	Register al(cpu,EAX,8), bl(cpu,R14,8), cl(cpu,EBX,8), dl(cpu,EDX,8);
	//Pointer-sized registers
	Register pax(cpu,EAX,pBits), pbx(cpu,R14,pBits), pcx(cpu,EBX,pBits), pdx(cpu,EDX,pBits), esp(cpu,ESP,pBits),
		pdi(cpu, R13, pBits), esi(cpu, R12, pBits), ebx(cpu, R14, pBits);
#else
	//32-bit registers
	Register eax(cpu,EAX), ebx(cpu,EBX), ecx(cpu,ECX), edx(cpu,EDX), ebp(cpu,EBP), edi(cpu,EDI);
	Register rarg(cpu, EDX);
	//8-bit registers
	Register al(cpu,EAX,8), bl(cpu,EBX,8), cl(cpu,ECX,8), dl(cpu,EDX,8);
	//Pointer-sized registers
	Register pax(cpu,EAX,pBits), pbx(cpu,EBX,pBits), pcx(cpu,ECX,pBits), pdx(cpu,EDX,pBits), esp(cpu,ESP,pBits),
		pdi(cpu, EDI, pBits), esi(cpu, ESI, pBits);
#endif


	//JIT FUNCTION ENTRY
	//==================
	//Push unmutable registers (these registers must retain their value after we leave our function)
	cpu.push(esi);
	cpu.push(edi);
	cpu.push(ebx);
	cpu.push(ebp);

	cpu.stackDepth += cpu.pushSize() * 4;

#ifdef JIT_64
	as<void*>(*esp+cpu.stackDepth) = cpu.intArg64(0, 0);
	as<void*>(*esp+cpu.stackDepth+cpu.pushSize()) = cpu.intArg64(1, 1);
	pax = cpu.intArg64(0, 0);
#else
	pax = as<void*>(*esp+cpu.stackDepth); //Register pointer
#endif

	//Function initialization {
	pdi = as<void*>(*pax+offsetof(asSVMRegisters,stackFramePointer)); //VM Frame pointer
	esi = as<void*>(*pax+offsetof(asSVMRegisters,stackPointer)); //VM Stack pointer
#ifdef JIT_64
	ebx = as<void*>(*pax+offsetof(asSVMRegisters,valueRegister)); //VM Temporary
#else
	ebx = *pax+offsetof(asSVMRegisters,valueRegister); //VM Temporary
	ebp = *pax+offsetof(asSVMRegisters,valueRegister)+4; //VM Temporary (upper half)
#endif
	pax = as<void*>(*esp+cpu.stackDepth+cpu.pushSize()); //Entry jump pointer
	//}

	//Jump to the section of the function we'll actually be executing this time
	cpu.jump(pax);

	//Function return {
	volatile byte* ret_pos = cpu.op;
	
	pax = as<void*>(*esp+cpu.stackDepth); //Register pointer
	as<void*>(*pax) = rarg; //Set the bytecode pointer based on our exit
	as<void*>(*pax+offsetof(asSVMRegisters,stackFramePointer)) = pdi; //Return the frame pointer
	as<void*>(*pax+offsetof(asSVMRegisters,stackPointer)) = esi; //Return the stack pointer
#ifdef JIT_64
	as<void*>(*pax+offsetof(asSVMRegisters,valueRegister)) = ebx; //Return the temporary
#else
	*pax+offsetof(asSVMRegisters,valueRegister) = ebx; //Return the temporary
	*pax+offsetof(asSVMRegisters,valueRegister)+4 = ebp; //  ...and the other half
#endif
	
	cpu.pop(ebp);
	cpu.pop(ebx);
	cpu.pop(edi);
	cpu.pop(esi);
	cpu.ret();
	//}

	auto Return = [&](bool expected) {
		//Set EDX to the bytecode pointer so the vm can be returned to the correct state
		rarg = (void*)pOp;
		cpu.jump(Jump,ret_pos);
		waitingForEntry = expected;
	};

	auto ReturnCondition = [&](JumpType condition) {
		rarg = (void*)pOp;
		cpu.jump(condition,ret_pos);
	};

	SystemCall sysCall(cpu, fpu, ReturnCondition, pOp, flags);

	volatile byte* script_ret = 0;
	auto ReturnFromScriptCall = [&]() {
		if(script_ret) {
			cpu.jump(Jump,script_ret);
		}
		else {
			script_ret = cpu.op;
			//The VM Registers are already in the correct state, so just do a simple return here
			cpu.pop(ebp);
			cpu.pop(ebx);
			cpu.pop(edi);
			cpu.pop(esi);
			cpu.ret();
		}
		waitingForEntry = true;
	};

	auto do_jump = [&](JumpType type) {
		asDWORD* bc = pOp + asBC_INTARG(pOp) + 2;
		auto& jmp = jumpTable[bc - start];
		if(jmp != 0) {
			//Jump to code that already exists
			cpu.jump(type, jmp);
		}
		else if(bc > pOp) {
			//Prep the jump for a future instruction
			futureJumps.insert(std::pair<asDWORD*,void*>(bc,cpu.prep_long_jump(type)));
		}
		else {
			//We can't handle this address, so generate a special return that does the jump ahead of time
			rarg = bc;
			cpu.jump(type, ret_pos);
		}
	};

	auto check_space = [&](unsigned bytes) {
		unsigned remaining = activePage->getFreeSize() - (cpu.op - byteStart);
		if(remaining < bytes) {
			CodePage* newPage = new CodePage(codePageSize);

			cpu.migrate(*activePage, *newPage);

			activePage = newPage;
			pages.insert(std::pair<asJITFunction,assembler::CodePage*>(*output,activePage));
			byteStart = (byte*)cpu.op;
		}
	};

	unsigned reservedPushBytes = 0;

	while(pOp < end) {
		asEBCInstr op = asEBCInstr(*(asBYTE*)pOp);
		auto firstJump = futureJumps.lower_bound(pOp), lastJump = futureJumps.upper_bound(pOp);

		if(waitingForEntry && op != asBC_JitEntry) {
			check_space(futureJumps.size() * (2 + sizeof(void*)*2));

			//Handle cases where we jump to code we can't directly handle
			if(firstJump != futureJumps.end() && firstJump->first == pOp) {
				for(auto i = firstJump; i != lastJump; ++i)
					cpu.end_long_jump(i->second);
				futureJumps.erase(firstJump, lastJump);
				check_space(32);
				Return(true);
			}

			pOp += toSize(op);
			continue;
		}

		if(cpu.op > activePage->getActivePage() + activePage->getFreeSize())
			throw "Page exceeded...";
		
		//Check for remaining space of at least 64 bytes (roughly 3 max-sized ops)
		// Do so before building jumps to save a jump when crossing pages
		check_space(64);

		jumpTable[pOp - start] = cpu.op;

		//Handle jumps to code we hadn't made yet
		if(firstJump != futureJumps.end() && firstJump->first == pOp) {
			for(auto i = firstJump; i != lastJump; ++i)
				cpu.end_long_jump(i->second);
			futureJumps.erase(firstJump, lastJump);
		}

		//Multi-op optimization - special cases where specific sets of ops serve a common purpose
		auto pNextOp = pOp + toSize(op);

		if(pNextOp < end) {
			auto nextOp = asEBCInstr(*(asBYTE*)pNextOp);

			auto pThirdOp = pNextOp + toSize(nextOp);
			if(pThirdOp < end) {
				auto thirdOp = asEBCInstr(*(asBYTE*)pThirdOp);

				switch(op) {
				case asBC_SetV8:
					if(thirdOp == asBC_CpyVtoV8 &&
						(nextOp == asBC_ADDd || nextOp == asBC_DIVd ||
						 nextOp == asBC_SUBd || nextOp == asBC_MULd)) {
						if(asBC_SWORDARG0(pOp) != asBC_SWORDARG2(pNextOp) || asBC_SWORDARG0(pOp) != asBC_SWORDARG0(pNextOp))
							break;

						//Optimize <Variable Double> <op>= <Constant Double>
						fpu.load_double(*edi-offset(pNextOp,1));

						MemAddress doubleConstant(cpu, &asBC_QWORDARG(pOp));

						switch(nextOp) {
						case asBC_ADDd:
							fpu.add_double(doubleConstant); break;
						case asBC_SUBd:
							fpu.sub_double(doubleConstant); break;
						case asBC_MULd:
							fpu.mult_double(doubleConstant); break;
						case asBC_DIVd:
							fpu.div_double(doubleConstant); break;
						}

						if(asBC_SWORDARG0(pOp) == asBC_SWORDARG1(pThirdOp)) {
							fpu.store_double(*edi-offset(pOp,0),false);
							fpu.store_double(*edi-offset(pThirdOp,0));
						
							pOp = pThirdOp + toSize(thirdOp);
						}
						else {
							fpu.store_double(*edi-offset(pOp,0));
						
							pOp = pThirdOp;
						}
					
						continue;
					}
					break;
				case asBC_SetV4:
					if(nextOp == asBC_SetV4 && thirdOp == asBC_SetV4 && asBC_DWORDARG(pOp) == asBC_DWORDARG(pNextOp) && asBC_DWORDARG(pNextOp) == asBC_DWORDARG(pThirdOp)) {
						//Optimize intializing 3 variables to the same value (often 0)
						if(asBC_DWORDARG(pOp) == 0)
							eax ^= eax;
						else
							eax = asBC_DWORDARG(pOp);
						*edi-offset(pOp,0) = eax;
						*edi-offset(pNextOp,0) = eax;
						*edi-offset(pThirdOp,0) = eax;

						pOp = pThirdOp + toSize(thirdOp);
						continue;
					}
					break;
				}
			}

			switch(op) {
			case asBC_SetV4:
				if(nextOp == asBC_SetV4 && asBC_DWORDARG(pOp) == asBC_DWORDARG(pNextOp)) {
					//Optimize intializing 2 variables to the same value (often 0)
					if(asBC_DWORDARG(pOp) == 0)
						eax ^= eax;
					else
						eax = asBC_DWORDARG(pOp);
					*edi-offset(pOp,0) = eax;
					*edi-offset(pNextOp,0) = eax;

					pOp = pThirdOp;
					continue;
				}
				break;
			case asBC_RDR4:
				if(nextOp == asBC_PshV4 && asBC_SWORDARG0(pOp) == asBC_SWORDARG0(pNextOp)) {
					//Optimize:
					//Store temporary int
					//Push stored temporary
					eax = *ebx;
					*edi-offset0 = eax;
					
					reservedPushBytes = findTotalPushBatchSize(pNextOp, end);
					esi -= reservedPushBytes;
					reservedPushBytes -= sizeof(asDWORD);
					*esi + reservedPushBytes = eax;

					pOp = pThirdOp;
					continue;
				}
				break;
			case asBC_PSF:
			case asBC_PshVPtr:
				if(reservedPushBytes == 0 && nextOp == asBC_COPY) {
					//Optimize:
					//Push Pointer
					//Copy Pointer
					//To:
					//Copy Pointer

					check_space(256);
					if(op == asBC_PSF)
						pax.copy_address(as<void*>(*edi-offset0));
					else //if(op == asBC_PshVPtr)
						pax = as<void*>(*edi-offset0);
					pcx = as<void*>(*esi);

					//Check for null pointers
					pax &= pax;
					void* test1 = cpu.prep_short_jump(Zero);
					pcx &= pcx;
					void* test2 = cpu.prep_short_jump(Zero);
					
					*esi = pax;

					cpu.call_cdecl((void*)memcpy,"rrc", &pax, &pcx, unsigned(asBC_WORDARG0(pNextOp))*4);
					void* skip_ret = cpu.prep_short_jump(Jump);
					//ERR
					cpu.end_short_jump(test1); cpu.end_short_jump(test2);
					Return(false);
					cpu.end_short_jump(skip_ret);

					pOp = pThirdOp;
					continue;
				}
				break;
			case asBC_CpyRtoV4:
				if(nextOp == asBC_CpyVtoV4 && offset(pOp,0) == offset(pNextOp,1)) {
					//Optimize
					//Copy Temp to Var X
					//Copy Var X to Var Y
					//To:
					//Copy Temp to Var X
					//Copy Temp to Var Y

					*edi-offset(pOp,0) = ebx;
					*edi-offset(pNextOp,0) = ebx;

					pOp = pThirdOp;
					continue;
				}
				break;
			case asBC_CpyVtoV4:
				if(nextOp == asBC_iTOf && offset(pOp,0) == offset(pNextOp,0)) {
					//Optimize:
					//Load integer
					//Convert integer to float in-place
					//To:
					//Load integer
					//Save float

					fpu.load_dword(*edi-offset(pOp,1));
					fpu.store_float(*edi-offset(pOp,0));

					pOp = pThirdOp;
					continue;
				}
				else if(nextOp == asBC_fTOd && offset(pOp,0) == offset(pNextOp,1)) {
					//Optimize:
					//Copy float
					//Convert float to double
					//To:
					//Copy float
					//Store double

					fpu.load_float(*edi-offset(pOp,1));
					fpu.store_float(*edi-offset(pOp,0),false);
					fpu.store_double(as<double>(*edi-offset(pNextOp,0)));

					pOp = pThirdOp;
					continue;
				}
				break;
			}
		}

		//Build ops
		switch(op) {
		case asBC_JitEntry:
			asBC_PTRARG(pOp) = (asPWORD)cpu.op;
			waitingForEntry = false;
			break;

		case asBC_POP:
			esi += asBC_WORDARG0(pOp) * sizeof(asDWORD);
			break;
		
		//Handle all pushes here by allocating all contiguous push memory at once
#define pushPrep(use) \
	if(reservedPushBytes == 0) {\
		reservedPushBytes = findTotalPushBatchSize(pOp, end);\
		esi -= reservedPushBytes;\
	}\
	reservedPushBytes -= use;

		case asBC_PUSH:
			pushPrep( asBC_WORDARG0(pOp) * sizeof(asDWORD) ); break;
		case asBC_PshC4:
			pushPrep(sizeof(asDWORD));
			*esi + reservedPushBytes = asBC_DWORDARG(pOp);
			break;
		case asBC_PshV4:
			pushPrep(sizeof(asDWORD));
			eax = *edi-offset0;
			*esi + reservedPushBytes = eax;
			break;
		case asBC_PSF:
			pushPrep(sizeof(void*));
			pax.copy_address(as<void*>(*edi-offset0));
			as<void*>(*esi + reservedPushBytes) = pax;
			break;
		case asBC_PshG4:
			pushPrep(sizeof(asDWORD));
			eax = MemAddress(cpu, (void*)asBC_PTRARG(pOp));
			*esi + reservedPushBytes = eax;
			break;
		case asBC_PshC8:
			{
				pushPrep(sizeof(asQWORD));
				asQWORD qword = asBC_QWORDARG(pOp);
#ifdef JIT_64
				as<asQWORD>(eax) = qword;
				as<asQWORD>(*esi + reservedPushBytes) = eax;
#else
				asDWORD* as_dword = (asDWORD*)&qword;
				*esi + reservedPushBytes+4 = as_dword[1];
				*esi + reservedPushBytes = as_dword[0];
#endif
			} break;
		case asBC_PshVPtr:
			pushPrep(sizeof(void*));
			pax = as<void*>(*edi-offset0);
			as<void*>(*esi + reservedPushBytes) = pax;
			break;
		case asBC_PshRPtr: 
			pushPrep(sizeof(void*));
			as<void*>(*esi + reservedPushBytes) = pbx;
			break;
		case asBC_PshNull:
			pushPrep(sizeof(void*));
			pax ^= pax;
			as<void*>(*esi + reservedPushBytes) = pax;
			break;
		case asBC_OBJTYPE:
			pushPrep(sizeof(void*));
			as<void*>(*esi + reservedPushBytes) = (void*)asBC_PTRARG(pOp);
			break;
		case asBC_TYPEID:
			pushPrep(sizeof(asDWORD));
			*esi + reservedPushBytes = asBC_DWORDARG(pOp);
			break;
		case asBC_FuncPtr:
			pushPrep(sizeof(void*));
			as<void*>(*esi + reservedPushBytes) = (void*)asBC_PTRARG(pOp);
			break;
		case asBC_PshV8:
			pushPrep(sizeof(asQWORD));
			cpu.setBitMode(64);
			(*esi + reservedPushBytes).direct_copy(*edi-offset0, eax);
			cpu.resetBitMode();
			break;
		case asBC_PGA:
			pushPrep(sizeof(void*));
			as<void*>(*esi + reservedPushBytes) = (void*)asBC_PTRARG(pOp);
			break;
		case asBC_VAR:
			pushPrep(sizeof(void*));
			as<void*>(*esi + reservedPushBytes) = (void*)asBC_SWORDARG0(pOp);
			break;

		////Now the normally-ordered ops
		case asBC_SwapPtr:
			pax = as<void*>(*esi);
			pax.swap(as<void*>(*esi+sizeof(void*)));
			as<void*>(*esi) = pax;
			break;
		case asBC_NOT:
			{
				pcx.copy_address(*edi-offset0);

				al = as<byte>(*pcx);
				al &= al;
				al.setIf(Zero);
				eax &= 0xff;
				*ecx = eax;
			} break;
		//case asBC_PshG4: //All pushes are handled above, near asBC_PshC4
		case asBC_LdGRdR4:
			pbx = (void*) asBC_PTRARG(pOp);
			eax = *pbx;
			*edi-offset0 = eax;
			break;
		case asBC_CALL:
			{
				pax = as<void*>(*esp+cpu.stackDepth);
				as<void*>(*pax + offsetof(asSVMRegisters,programPointer)) = pOp+2;
				as<void*>(*pax + offsetof(asSVMRegisters,stackPointer)) = esi;
				pax = as<void*>(*pax + offsetof(asSVMRegisters,ctx));

				cpu.call_stdcall((void*)callScriptFunction,"rc",
					&pax,
					(asCScriptFunction*)function->GetEngine()->GetFunctionById(asBC_INTARG(pOp))
					);
				ReturnFromScriptCall();
			} break;
		case asBC_RET:
			{
				check_space(128);
				pax = as<void*>(*esp+cpu.stackDepth);
				pax = as<void*>(*pax + offsetof(asSVMRegisters,ctx));
				cpu.call_stdcall((void*)popStackIfNotEmpty,"r",&pax);

				//Normally we have to unload our full state, but in a return, the bytecode, frame, and stack pointer are already where they need to be
				pdx = as<void*>(*esp+cpu.stackDepth); //Register pointer
#ifdef JIT_64
				as<void*>(*pdx+offsetof(asSVMRegisters,valueRegister)) = pbx; //VM Temporary
#else
				*pdx+offsetof(asSVMRegisters,valueRegister) = ebx; //VM Temporary
				*pdx+offsetof(asSVMRegisters,valueRegister)+4 = ebp; //VM Temporary (upper half)
#endif
				as<void*>(*pdx+offsetof(asSVMRegisters,stackPointer)) += asBC_WORDARG0(pOp)*sizeof(asDWORD);
				cpu.pop(ebp);
				cpu.pop(ebx);
				cpu.pop(edi);
				cpu.pop(esi);
				cpu.ret();
				waitingForEntry = true;
			}
			break;
		case asBC_JMP:
			do_jump(Jump);
			break;

		case asBC_JZ:
			bl &= bl; do_jump(Zero); break;
		case asBC_JNZ:
			bl &= bl; do_jump(NotZero); break;
		case asBC_JS:
			bl &= bl; do_jump(Sign); break;
		case asBC_JNS:
			bl &= bl; do_jump(NotSign); break;
		case asBC_JP:
			bl == 0; do_jump(Greater); break;
		case asBC_JNP:
			bl == 0; do_jump(LessOrEqual); break;

		case asBC_TZ:
			bl &= bl; ebx.setIf(Zero); break;
		case asBC_TNZ:
			bl &= bl; ebx.setIf(NotZero); break;
		case asBC_TS:
			bl &= bl; ebx.setIf(Sign); break;
		case asBC_TNS:
			bl &= bl; ebx.setIf(NotSign); break;
		case asBC_TP:
			bl == 0; ebx.setIf(Greater); break;
		case asBC_TNP:
			bl == 0; ebx.setIf(LessOrEqual); break;

		case asBC_NEGi:
			-(*edi-offset0);
			break;
		case asBC_NEGf:
			fpu.load_float(*edi-offset0);
			fpu.negate();
			fpu.store_float(*edi-offset0);
			break;
		case asBC_NEGd:
			fpu.load_double(*edi-offset0);
			fpu.negate();
			fpu.store_double(*edi-offset0);
			break;
		case asBC_INCi16:
			++as<short>(*ebx);
			break;
		case asBC_INCi8:
			++as<char>(*ebx);
			break;
		case asBC_DECi16:
			--as<short>(*ebx);
			break;
		case asBC_DECi8:
			--as<char>(*ebx);
			break;
		case asBC_INCi:
			++*ebx;
			break;
		case asBC_DECi:
			--*ebx;
			break;
		case asBC_INCf:
			fpu.load_const_1();
			fpu.add_float(*ebx);
			fpu.store_float(*ebx);
			break;
		case asBC_DECf:
			fpu.load_const_1();
			fpu.negate();
			fpu.add_float(*ebx);
			fpu.store_float(*ebx);
			break;
		case asBC_INCd:
			fpu.load_const_1();
			fpu.add_double(*ebx);
			fpu.store_double(*ebx);
			break;
		case asBC_DECd:
			fpu.load_const_1();
			fpu.negate();
			fpu.add_double(*ebx);
			fpu.store_double(*ebx);
			break;
		case asBC_IncVi:
			++(*edi-offset0);
			break;
		case asBC_DecVi:
			--(*edi-offset0);
			break;
		case asBC_BNOT:
			~(*edi-offset0);
			break;
		case asBC_BAND:
			eax = *edi-offset1;
			eax &= *edi-offset2;
			*edi-offset0 = eax;
			break;
		case asBC_BOR:
			eax = *edi-offset1;
			eax |= *edi-offset2;
			*edi-offset0 = eax;
			break;
		case asBC_BXOR:
			eax = *edi-offset1;
			eax ^= *edi-offset2;
			*edi-offset0 = eax;
			break;
		case asBC_BSLL: {
			Register c(cpu, ECX);
			eax = *edi-offset1;
			c = *edi-offset2;
			eax <<= c;
			*edi-offset0 = eax;
			} break;
		case asBC_BSRL: {
			Register c(cpu, ECX);
			eax = *edi-offset1;
			c = *edi-offset2;
			eax.rightshift_logical(c);
			*edi-offset0 = eax;
			} break;
		case asBC_BSRA: {
			Register c(cpu, ECX);
			eax = *edi-offset1;
			c = *edi-offset2;
			eax >>= c;
			*edi-offset0 = eax;
			} break;
		case asBC_COPY:
			{
				check_space(128);
				pax = as<void*>(*esi);
				esi += sizeof(void*);
				pcx = as<void*>(*esi);

				//Check for null pointers
				pax &= pax;
				void* test1 = cpu.prep_short_jump(Zero);
				pcx &= pcx;
				void* test2 = cpu.prep_short_jump(Zero);
				
				*esi = pax;

				cpu.call_cdecl((void*)memcpy,"rrc", &pax, &pcx, unsigned(asBC_WORDARG0(pOp))*4);
				void* skip_ret = cpu.prep_short_jump(Jump);
				//ERR
				cpu.end_short_jump(test1); cpu.end_short_jump(test2);
				Return(false);
				cpu.end_short_jump(skip_ret);
			} break;
		//case asBC_PshC8: //All pushes are handled above, near asBC_PshC4
		//case asBC_PshVPtr:
		case asBC_RDSPtr:
			pax = as<void*>(*esi);
			pax = as<void*>(*pax);
			as<void*>(*esi) = pax;
			break;
		case asBC_CMPd:
			{
				fpu.load_double(*edi-offset1);
				fpu.load_double(*edi-offset0);
				fpu.compare_toCPU(FPU_1);

				bl.setIf(Above);
				auto t2 = cpu.prep_short_jump(NotCarry);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);

				fpu.pop();
			} break;
		case asBC_CMPu:
			{
				eax = *edi-offset0;
				eax == *edi-offset1;

				bl.setIf(Above);
				auto t2 = cpu.prep_short_jump(NotBelow);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);
			} break;
		case asBC_CMPf:
			{
				fpu.load_float(*edi-offset1);
				fpu.load_float(*edi-offset0);
				fpu.compare_toCPU(FPU_1);

				bl.setIf(Above);
				auto t2 = cpu.prep_short_jump(NotCarry);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);

				fpu.pop();
			} break;
		case asBC_CMPi:
			{
				eax = *edi-offset0;
				eax == *edi-offset1;

				bl.setIf(Greater);
				auto t2 = cpu.prep_short_jump(GreaterOrEqual);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);
			} break;
		case asBC_CMPIi:
			{
				eax = *edi-offset0;
				eax == asBC_DWORDARG(pOp);

				bl.setIf(Greater);
				auto t2 = cpu.prep_short_jump(GreaterOrEqual);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);
			} break;
		case asBC_CMPIf:
			{
				fpu.load_float(MemAddress(cpu,&asBC_FLOATARG(pOp)));
				fpu.load_float(*edi-offset0);
				fpu.compare_toCPU(FPU_1);

				bl.setIf(Above);
				auto t2 = cpu.prep_short_jump(NotCarry);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);

				fpu.pop();
			} break;
		case asBC_CMPIu:
			{
				eax = *edi-offset0;
				eax == asBC_DWORDARG(pOp);

				bl.setIf(Above);
				auto t2 = cpu.prep_short_jump(NotBelow);
				~bl; //0xff if < 0
				cpu.end_short_jump(t2);
			} break;
		case asBC_JMPP:
			if((flags & JIT_NO_SWITCHES) == 0) {
				tableInUse = true;

				pax = (void*)(jumpTable + ((pOp + 1) - start));

				pdx.copy_expanding(as<int>(*edi - offset0));
				pdx += pdx;

				pcx = as<void*>(*pax + pdx*sizeof(void*));

				//Check for a pointer in the jump table to executable code
				pcx &= pcx;
				auto handled_jump = cpu.prep_short_jump(NotZero);

				//Copy the offsetted pointer to edx and return
				pcx = (void*)(pOp + 1);
				rarg.copy_address(*pcx + pdx*4);
				cpu.jump(Jump,ret_pos);

				cpu.end_short_jump(handled_jump);
				cpu.jump(pcx);
			}
			else {
				Return(true);
			}
			break;
		case asBC_PopRPtr:
			pbx = as<void*>(*esi);
			esi += sizeof(void*);
			break;
		//case asBC_PshRPtr: //All pushes are handled above, near asBC_PshC4
		case asBC_STR:
			{
				const asCString &str = ((asCScriptEngine*)function->GetEngine())->GetConstantString(asBC_WORDARG0(pOp));
				esi -= sizeof(void*) + sizeof(asDWORD);
				as<void*>(*esi + sizeof(asDWORD)) = (void*)str.AddressOf();
				as<asDWORD>(*esi) = (asDWORD)str.GetLength();
			} break;
		case asBC_CALLSYS:
			{
				check_space(512);
				asCScriptFunction* func = (asCScriptFunction*)function->GetEngine()->GetFunctionById(asBC_INTARG(pOp));

				sysCall.callSystemFunction(func);
			} break;
		case asBC_CALLBND:
			{
				pax = as<void*>(*esp+cpu.stackDepth);
				as<void*>(*pax + offsetof(asSVMRegisters,programPointer)) = pOp+2;
				as<void*>(*pax + offsetof(asSVMRegisters,stackPointer)) = esi;
				pax = as<void*>(*pax + offsetof(asSVMRegisters,ctx));

				cpu.call_stdcall((void*)callBoundFunction,"rc",
					&pax,
					(unsigned int)asBC_INTARG(pOp));
				pax &= pax;
				auto p2 = cpu.prep_short_jump(Zero);
				Return(false);
				cpu.end_short_jump(p2);
				ReturnFromScriptCall();
			} break;
		case asBC_SUSPEND:
			if(flags & JIT_NO_SUSPEND) {
				//Do nothing
			}
			else {
				pax = as<void*>(*esp + cpu.stackDepth);

				//Check if we should suspend
				cl = as<byte>(*pax+offsetof(asSVMRegisters,doProcessSuspend));
				cl &= cl;
				auto skip = cpu.prep_short_jump(Zero);
				
				as<void*>(*pax + offsetof(asSVMRegisters,programPointer)) = pOp;
				as<void*>(*pax + offsetof(asSVMRegisters,stackPointer)) = esi;

				pdx = as<void*>(*eax + offsetof(asSVMRegisters,ctx));
				cpu.call_stdcall((void*)doSuspend, "r", &pdx);

				//If doSuspend return true, return to AngelScript for a suspension
				rarg = (void*)pOp;
				al &= al;
				cpu.jump(NotZero, ret_pos);
				
				cpu.end_short_jump(skip);
			}
			break;
		case asBC_ALLOC:
			{
				check_space(512);
				asCObjectType *objType = (asCObjectType*)(size_t)asBC_PTRARG(pOp);
				int func = asBC_INTARG(pOp+AS_PTR_SIZE);

				if(objType->flags & asOBJ_SCRIPT_OBJECT) {
					asIScriptEngine* engine = function->GetEngine();
					asCScriptFunction* f = ((asCScriptEngine*)engine)->GetScriptFunction(func);

					MemAddress regPointer(*esp + cpu.stackDepth);

					cpu.call_stdcall((void*)allocScriptObject,"cccm",objType,f,engine,&regPointer);

					ReturnFromScriptCall();
				}
				else {
					cpu.call_stdcall((void*)engineAlloc,"cc",
						(asCScriptEngine*)function->GetEngine(),
						objType);

					if( func ) {
						cpu.push(pax); cpu.stackDepth += sizeof(void*);						
						auto pFunc = (asCScriptFunction*)function->GetEngine()->GetFunctionById(func);

						pcx = pax;
						sysCall.callSystemFunction(pFunc, &pcx);

						cpu.stackDepth -= sizeof(void*);
						cpu.pop(pax);
					}

					//Pop pointer destination from vm stack
					pcx = as<void*>(*esi);
					esi += sizeof(void*);

					//Set it if not zero
					pcx &= pcx;
					auto p = cpu.prep_short_jump(Zero);
					as<void*>(*pcx) = pax;
					cpu.end_short_jump(p);
				}
			} break;
		case asBC_FREE:
			{
			asCObjectType *objType = (asCObjectType*)(size_t)asBC_PTRARG(pOp);

			if(!(objType->flags & asOBJ_REF) || !(objType->flags & asOBJ_NOCOUNT)) { //Only do FREE on non-reference types, or reference types without fake reference counting
				check_space(128);
				asSTypeBehaviour *beh = &objType->beh;

				//Check the pointer to see if it's already zero
				pcx = as<void*>(*edi-offset0);
				pcx &= pcx;
				auto p = cpu.prep_short_jump(Zero);

				//Copy over registers to the vm in case the called functions observe the call stack
				if((flags & JIT_ALLOC_SIMPLE) == 0) {
					cpu.setBitMode(sizeof(void*)*8);
					pdx = *esp+cpu.stackDepth;
					*pdx + offsetof(asSVMRegisters,programPointer) = pOp;
					*pdx + offsetof(asSVMRegisters,stackPointer) = esi;
					cpu.resetBitMode();
				}

				if(beh->release) {
					cpu.call_stdcall((void*)engineRelease,"crc",
						(asCScriptEngine*)function->GetEngine(),
						&pcx,
						(asCScriptFunction*)function->GetEngine()->GetFunctionById(beh->release) );
				}
				else if(beh->destruct) {
					cpu.call_stdcall((void*)engineDestroyFree,"crc",
						(asCScriptEngine*)function->GetEngine(),
						&pcx,
						(asCScriptFunction*)function->GetEngine()->GetFunctionById(beh->destruct) );
				}
				else {
					cpu.call_stdcall((void*)engineFree,"cr",
						(asCScriptEngine*)function->GetEngine(),
						&pcx);
				}

				as<void*>(*edi-offset0) = nullptr;
				cpu.end_short_jump(p);
			}
			}break;
		case asBC_LOADOBJ:
			{
				cpu.setBitMode(sizeof(void*)*8);
				eax = *edi-offset0;
				edx = *esp+cpu.stackDepth;
				*edx+offsetof(asSVMRegisters,objectType) = nullptr;
				*edx+offsetof(asSVMRegisters,objectRegister) = eax;
				*edi-offset0 = nullptr;
				cpu.resetBitMode();
			} break;
		case asBC_STOREOBJ:
			{
				cpu.setBitMode(sizeof(void*) * 8);
				edx = *esp+cpu.stackDepth;
				(*edi-offset0).direct_copy( (*edx+offsetof(asSVMRegisters,objectRegister)), eax);
				*edx+offsetof(asSVMRegisters,objectRegister) = nullptr;
				cpu.resetBitMode();
			} break;
		case asBC_GETOBJ:
			{
				pax.copy_address(*esi+offset0);

				pdx = as<asDWORD>(*eax); //-Offset
				-pdx;

				pcx.copy_address(*edi+pdx*4);

				as<void*>(*pax).direct_copy(as<void*>(*pcx), pdx);
				as<void*>(*pcx) = nullptr;
			} break;
		case asBC_REFCPY:
			{
			asCObjectType *objType = (asCObjectType*)(size_t)asBC_PTRARG(pOp);

			if(objType->flags & asOBJ_NOCOUNT) {
				pax = as<void*>(*esi);
				esi += sizeof(void*);
				pcx = as<void*>(*esi);
				as<void*>(*pax) = pcx;
			}
			else {
				check_space(512);
				//Copy over registers to the vm in case the called functions observe the call stack
				pcx = as<void*>(*esp+cpu.stackDepth);
				as<void*>(*pcx + offsetof(asSVMRegisters,programPointer)) = pOp;
				as<void*>(*pcx + offsetof(asSVMRegisters,stackPointer)) = esi;

				asSTypeBehaviour *beh = &objType->beh;

				cpu.push(as<void*>(*esi));
				esi += sizeof(void*);
				pcx = as<void*>(*esi);

				pcx &= pcx; cpu.push(pcx);
				cpu.stackDepth += cpu.pushSize() * 2;
				auto prev = cpu.prep_short_jump(Zero);
				cpu.call_stdcall((void*)engineCallMethod,"crc",
					(asCScriptEngine*)function->GetEngine(),
					&pcx,
					(asCScriptFunction*)function->GetEngine()->GetFunctionById(beh->addref) );
				cpu.end_short_jump(prev);

				pcx = as<void*>(*esp+sizeof(void*));
				pcx = as<void*>(*pcx);
				pcx &= pcx;
				auto dest = cpu.prep_short_jump(Zero);
				cpu.call_stdcall((void*)engineCallMethod,"crc",
					(asCScriptEngine*)function->GetEngine(),
					&pcx,
					(asCScriptFunction*)function->GetEngine()->GetFunctionById(beh->release) );
				cpu.end_short_jump(dest);

				cpu.pop(pcx); cpu.pop(pdx);
				cpu.stackDepth -= cpu.pushSize() * 2;
				as<void*>(*pdx) = pcx;
			}
			}break;
		case asBC_CHKREF:
			{
				pax = as<void*>(*esi);
				pax &= pax;
				auto p = cpu.prep_short_jump(NotZero);
				Return(false);
				cpu.end_short_jump(p);
			} break;
		case asBC_GETOBJREF:
			pax.copy_address(*esi + (asBC_WORDARG0(pOp)*sizeof(asDWORD)));

			pcx = as<void*>(*pax); //-Offset
			-pcx;

			as<void*>(*pax).direct_copy(as<void*>(*pdi+pcx*sizeof(asDWORD)), pdx);
			break;
		case asBC_GETREF:
			pax.copy_address(*esi + (asBC_WORDARG0(pOp)*sizeof(asDWORD)));

			pcx = as<void*>(*pax); //-Offset
			-pcx;

			pcx.copy_address(*pdi+pcx*sizeof(asDWORD));
			as<void*>(*pax) = pcx;
			break;
		//case asBC_PshNull: //All pushes are handled above, near asBC_PshC4
		case asBC_ClrVPtr:
			pax ^= pax;
			as<void*>(*edi-offset0) = pax;
			break;
		//case asBC_OBJTYPE: //All pushes are handled above, near asBC_PshC4
		//case asBC_TYPEID:
		case asBC_SetV1: //V1 and V2 are identical on little-endian processors
		case asBC_SetV2:
		case asBC_SetV4:
			*edi-offset0 = asBC_DWORDARG(pOp);
			break;
		case asBC_SetV8:
			{
#ifdef JIT_64
			pax = asBC_QWORDARG(pOp);
			as<asQWORD>(*edi-offset0) = pax;
#else
			asQWORD* input = &asBC_QWORDARG(pOp);
			asDWORD* data = (asDWORD*)input;
			*edi-offset0+4 = *(data+1);
			*edi-offset0 = *data;
#endif
			} break;
		//case asBC_ADDSi:
			//*esi += asBC_SWORDARG0(pOp);
			//break;

		case asBC_CpyVtoV4:
			as<asDWORD>(*edi-offset0).direct_copy(as<asDWORD>(*edi-offset1), eax);
			break;
		case asBC_CpyVtoV8:
			as<long long>(*edi-offset0).direct_copy(as<long long>(*edi-offset1), eax);
			break;

		case asBC_CpyVtoR4:
			ebx = *edi - offset0;
			break;
		case asBC_CpyVtoR8:
#ifdef JIT_64
			ebx = as<void*>(*edi-offset0);
#else
			ebx = *edi-offset0;
			ebp = *edi-offset0+4;
#endif
			break;

		case asBC_CpyVtoG4:
			eax = *edi-offset0;
			MemAddress(cpu, (void*)asBC_PTRARG(pOp)) = eax;
			break;

		case asBC_CpyRtoV4:
			as<unsigned>(*edi-offset0) = as<unsigned>(ebx);
			break;
		case asBC_CpyRtoV8:
#ifdef JIT_64
			as<void*>(*edi-offset0) = pbx;
#else
			*edi-offset0 = ebx;
			*edi-offset0+4 = ebp;
#endif
			break;

		case asBC_CpyGtoV4:
			eax = MemAddress(cpu, (void*)asBC_PTRARG(pOp));
			*edi-offset0 = eax;
			break;

		case asBC_WRTV1:
			cpu.setBitMode(8);
			(*ebx).direct_copy(*edi-offset0, eax);
			cpu.resetBitMode();
			break;
		case asBC_WRTV2:
			cpu.setBitMode(16);
			(*ebx).direct_copy(*edi-offset0, eax);
			cpu.resetBitMode();;
			break;
		case asBC_WRTV4:
			cpu.setBitMode(32);
			(*pbx).direct_copy(*edi-offset0, eax);
			cpu.resetBitMode();
			break;
		case asBC_WRTV8:
			cpu.setBitMode(64);
			(*ebx).direct_copy(*edi-offset0, eax);
			cpu.resetBitMode();
			break;

		case asBC_RDR1:
			eax = *ebx;
			eax &= 0x000000ff;
			*edi-offset0 = eax;
			break;
		case asBC_RDR2:
			eax = *ebx;
			eax &= 0x0000ffff;
			*edi-offset0 = eax;
			break;
		case asBC_RDR4:
			as<asDWORD>(*edi-offset0).direct_copy(as<asDWORD>(*ebx), eax); break;
		case asBC_RDR8:
			as<asQWORD>(*edi-offset0).direct_copy(as<asQWORD>(*ebx), eax); break;

		case asBC_LDG:
			pbx = (void*)asBC_PTRARG(pOp);
			break;
		case asBC_LDV:
			pbx.copy_address(*edi-offset0);
			break;
		//case asBC_PGA: //All pushes are handled above, near asBC_PshC4
		case asBC_CmpPtr:
			{
			pax = as<void*>(*edi-offset0);
			pax == as<void*>(*edi-offset1);

			bl.setIf(Above);
			auto t2 = cpu.prep_short_jump(NotBelow);
			~bl; //0xff if < 0
			cpu.end_short_jump(t2);
			}
			break;
		//case asBC_VAR: //All pushes are handled above, near asBC_PshC4
		case asBC_sbTOi:
			eax.copy_expanding(as<char>(*edi-offset0));
			*edi-offset0 = eax;
			break;
		case asBC_swTOi:
			eax.copy_expanding(as<short>(*edi-offset0));
			*edi-offset0 = eax;
			break;
		case asBC_ubTOi:
			*edi-offset0 &= 0xff;
			break;
		case asBC_uwTOi:
			*edi-offset0 &= 0xffff;
			break;
		case asBC_ADDi:
			eax = *edi-offset1;
			eax += *edi-offset2;
			*edi-offset0 = eax;
			break;
		case asBC_SUBi:
			eax = *edi-offset1;
			eax -= *edi-offset2;
			*edi-offset0 = eax;
			break;
		case asBC_MULi:
			eax = *edi-offset1;
			eax *= *edi-offset2;
			*edi-offset0 = eax;
			break;
		case asBC_DIVi:
			ecx = *edi-offset2;

			eax = ecx;
			eax &= eax;
			{
			void* zero_test = cpu.prep_short_jump(NotZero);
			Return(false);
			cpu.end_short_jump(zero_test);
			}

			eax = *edi-offset1;
			edx ^= edx;

			{
			eax == 0;
			auto notSigned = cpu.prep_short_jump(NotSign);
			~edx;
			cpu.end_short_jump(notSigned);
			}

			ecx.divide_signed();

			*edi-offset0 = eax;
			break;
		case asBC_MODi:
			ecx = *edi-offset2;

			eax = ecx;
			eax &= eax;
			{
			void* zero_test = cpu.prep_short_jump(NotZero);
			Return(false);
			cpu.end_short_jump(zero_test);
			}

			eax = *edi-offset1;
			edx ^= edx;

			{
			eax == 0;
			auto notSigned = cpu.prep_short_jump(NotSign);
			~edx;
			cpu.end_short_jump(notSigned);
			}

			ecx.divide_signed();

			*edi-offset0 = edx;
			break;
		case asBC_ADDf:
			fpu.load_float(*edi-offset1);
			fpu.add_float(*edi-offset2);
			fpu.store_float(*edi-offset0);
			break;
		case asBC_SUBf:
			fpu.load_float(*edi-offset1);
			fpu.sub_float(*edi-offset2);
			fpu.store_float(*edi-offset0);
			break;
		case asBC_MULf:
			fpu.load_float(*edi-offset1);
			fpu.mult_float(*edi-offset2);
			fpu.store_float(*edi-offset0);
			break;
		case asBC_DIVf:
			fpu.load_float(*edi-offset1);
			fpu.div_float(*edi-offset2);
			fpu.store_float(*edi-offset0);
			break;
		case asBC_MODf:
			ecx.copy_address(*edi-offset1);
			eax.copy_address(*edi-offset2);
			cpu.call_stdcall((void*)fmod_wrapper_f,"rr",&ecx,&eax);
#ifdef JIT_64
			*edi-offset0 = cpu.floatReturn64();
#else
			fpu.store_float(*edi-offset0);
#endif
			break;
		case asBC_ADDd:
			fpu.load_double(*edi-offset1);
			fpu.add_double(*edi-offset2);
			fpu.store_double(*edi-offset0);
			break;
		case asBC_SUBd:
			fpu.load_double(*edi-offset1);
			fpu.sub_double(*edi-offset2);
			fpu.store_double(*edi-offset0);
			break;
		case asBC_MULd:
			fpu.load_double(*edi-offset1);
			fpu.mult_double(*edi-offset2);
			fpu.store_double(*edi-offset0);
			break;
		case asBC_DIVd:
			//TODO: AngelScript considers division by 0 an error, should we?
			fpu.load_double(*edi-offset1);
			fpu.div_double(*edi-offset2);
			fpu.store_double(*edi-offset0);
			break;
		case asBC_MODd:
			pcx.copy_address(*edi-offset1);
			pax.copy_address(*edi-offset2);
			cpu.call_stdcall((void*)fmod_wrapper,"rr",&pcx,&pax);
#ifdef JIT_64
			as<double>(*edi-offset0) = cpu.floatReturn64();
#else
			fpu.store_double(*edi-offset0);
#endif
			break;
		case asBC_ADDIi:
			(*edi-offset0).direct_copy(*edi-offset1,eax);
			*edi-offset0 += asBC_INTARG(pOp+1);
			break;
		case asBC_SUBIi:
			(*edi-offset0).direct_copy(*edi-offset1,eax);
			*edi-offset0 -= asBC_INTARG(pOp+1);
			break;
		case asBC_MULIi:
			eax.multiply_signed(*edi-offset1,asBC_INTARG(pOp+1));
			*edi-offset0 = eax;
			break;
		case asBC_ADDIf:
			fpu.load_float(*edi-offset1);
			fpu.add_float( MemAddress(cpu,&asBC_FLOATARG(pOp+1)) );
			fpu.store_float(*edi-offset0);
			break;
		case asBC_SUBIf:
			fpu.load_float(*edi-offset1);
			fpu.sub_float( MemAddress(cpu,&asBC_FLOATARG(pOp+1)) );
			fpu.store_float(*edi-offset0);
			break;
		case asBC_MULIf:
			fpu.load_float(*edi-offset1);
			fpu.mult_float( MemAddress(cpu,&asBC_FLOATARG(pOp+1)) );
			fpu.store_float(*edi-offset0);
			break;
		case asBC_SetG4:
			MemAddress(cpu,(void*)asBC_PTRARG(pOp)) = asBC_DWORDARG(pOp+AS_PTR_SIZE);
			break;
		case asBC_ChkRefS:
			//Return if *(*esi) == 0
			pax = as<void*>(*esi);
			eax = as<int>(*pax);
			eax &= eax;
			rarg = (void*)pOp;
			cpu.jump(Zero,ret_pos);
			break;
		case asBC_ChkNullV:
			//Return if (*edi-offset0) == 0
			eax = *edi-offset0;
			eax &= eax;
			rarg = (void*)pOp;
			cpu.jump(Zero,ret_pos);
			break;
		case asBC_CALLINTF:
			{
				cpu.setBitMode(sizeof(void*)*8);
				ecx = *esp+cpu.stackDepth;
				*ecx + offsetof(asSVMRegisters,programPointer) = pOp+2;
				*ecx + offsetof(asSVMRegisters,stackPointer) = esi;
				cpu.resetBitMode();

				MemAddress ctxPtr( as<void*>(*ecx + offsetof(asSVMRegisters,ctx)) );

				cpu.call_stdcall((void*)callInterfaceMethod,"mc",
					&ctxPtr,
					(asCScriptFunction*)function->GetEngine()->GetFunctionById(asBC_INTARG(pOp))
					);
				ReturnFromScriptCall();
			} break;
		//asBC_SetV1 and asBC_SetV2 are aliased to asBC_SetV4
		case asBC_Cast: //Can't handle casts (script call)
			{
				pcx = as<void*>(*esi);
				pcx &= pcx;
				auto toEnd1 = cpu.prep_short_jump(Zero);
				pcx = as<void*>(*ecx);
				pcx &= pcx;
				auto toEnd2 = cpu.prep_short_jump(Zero);
				
				asCObjectType *to = ((asCScriptEngine*)function->GetEngine())->GetObjectTypeFromTypeId(asBC_DWORDARG(pOp));
				cpu.call_stdcall((void*)castObject,"rc",&ecx,to);
				pax &= pax;
				auto toEnd3 = cpu.prep_short_jump(Zero);

				pcx = as<void*>(*esp+cpu.stackDepth);
				as<void*>(*pcx + offsetof(asSVMRegisters,objectRegister)) = pax;
				
				cpu.end_short_jump(toEnd1);
				cpu.end_short_jump(toEnd2);
				cpu.end_short_jump(toEnd3);
				esi += sizeof(void*);
			} break;

		case asBC_iTOb:
			*edi-offset0 &= 0xff;
			break;
		case asBC_iTOw:
			*edi-offset0 &= 0xffff;
			break;

#ifdef _MSC_VER
#define cast(f,t) {\
	void* func = (void*)(void (*)(f*,t*))(directConvert<f,t>);\
	pax.copy_address(*edi-offset0);\
	if(sizeof(f) != sizeof(t))\
		{ pcx.copy_address(*edi-offset1); cpu.call_stdcall(func,"rr",&pcx,&pax); }\
	else\
		cpu.call_stdcall(func,"rr",&pax,&pax);\
	}
#else
#define cast(f,t) {\
	void* func = (void*)(void (*)(f*,t*))(directConvert<f,t>);\
	pax.copy_address(*edi-offset0);\
	if(sizeof(f) != sizeof(t))\
		{ pcx.copy_address(*edi-offset1); cpu.call_cdecl(func,"rr",&pcx,&pax); }\
	else\
		cpu.call_cdecl(func,"rr",&pax,&pax);\
	}
#endif

		////All type conversions of QWORD to/from DWORD and Float to/from Int are here
		case asBC_iTOf:
			fpu.load_dword(*edi-offset0);
			fpu.store_float(*edi-offset0);
			break;
		case asBC_fTOi:
			cast(float,int); break;
		case asBC_uTOf:
			cast(unsigned, float); break;
		case asBC_fTOu:
			cast(float, unsigned); break;
		case asBC_dTOi:
			cast(double,int); break;
		case asBC_dTOu:
			cast(double, unsigned); break;
		case asBC_dTOf:
			fpu.load_double(*edi-offset1);
			fpu.store_float(*edi-offset0);
			break;
		case asBC_iTOd:
			fpu.load_dword(*edi-offset1);
			fpu.store_double(*edi-offset0);
			break;
		case asBC_uTOd:
			cast(unsigned, double); break;
		case asBC_fTOd:
			fpu.load_float(*edi-offset1);
			fpu.store_double(*edi-offset0);
			break;
		case asBC_i64TOi:
			cast(long long, int) break;
		case asBC_uTOi64:
			cast(unsigned int, long long) break;
		case asBC_iTOi64:
			cast(int, long long) break;
		case asBC_fTOi64:
			cast(float, long long) break;
		case asBC_fTOu64:
			cast(float, unsigned long long) break;
		case asBC_i64TOf:
			cast(long long, float) break;
		case asBC_u64TOf:
			cast(unsigned long long, float) break;
		case asBC_dTOi64:
			cast(double, long long) break;
		case asBC_dTOu64:
			cast(double, unsigned long long) break;
		case asBC_i64TOd:
			cast(long long, double) break;
		case asBC_u64TOd:
			cast(unsigned long long, double) break;

		case asBC_NEGi64:
			-as<long long>(*edi-offset0);
			break;
		case asBC_INCi64:
			++as<long long>(*ebx);
			break;
		case asBC_DECi64:
			--as<long long>(*ebx);
			break;
		case asBC_BNOT64:
			~as<long long>(*edi-offset0);
			break;
		case asBC_ADDi64:
			{
#ifdef JIT_64
			pax = as<int64_t>(*edi-offset1);
			pax += as<int64_t>(*edi-offset2);
			as<int64_t>(*edi-offset0) = pax;
#else
			eax = *edi-offset1;
			eax += *edi-offset2;
			*edi-offset0 = eax;
			eax = *edi-offset1+4;
			auto p = cpu.prep_short_jump(NotOverflow);
			++eax;
			cpu.end_short_jump(p);
			eax += *edi-offset2+4;
			*edi-offset0+4 = eax;
#endif
			} break;
		case asBC_SUBi64:
			{
#ifdef JIT_64
			pax = as<int64_t>(*edi-offset1);
			pax -= as<int64_t>(*edi-offset2);
			as<int64_t>(*edi-offset0) = pax;
#else
			eax = *edi-offset1;
			eax -= *edi-offset2;
			*edi-offset0 = eax;
			eax = *edi-offset1+4;
			auto p = cpu.prep_short_jump(NotOverflow);
			--eax;
			cpu.end_short_jump(p);
			eax -= *edi-offset2+4;
			*edi-offset0+4 = eax;
#endif
			} break;
		case asBC_MULi64:
#ifdef JIT_64
			pax = as<int64_t>(*edi-offset1);
			pax *= as<int64_t>(*edi-offset2);
			as<int64_t>(*edi-offset0) = pax;
#else
			ecx.copy_address(*edi-offset1);
			edx.copy_address(*edi-offset2);
			eax.copy_address(*edi-offset0);
			cpu.call_stdcall((void*)i64_mul,"rrr",&ecx,&edx,&eax);
#endif
			break;
		case asBC_DIVi64:
			pcx.copy_address(*edi-offset1);
			rarg.copy_address(*edi-offset2);
			pax.copy_address(*edi-offset0);
			cpu.call_stdcall((void*)i64_div,"rrr",&pcx,&rarg,&pax);
			break;
		case asBC_MODi64:
			pcx.copy_address(*edi-offset1);
			rarg.copy_address(*edi-offset2);
			pax.copy_address(*edi-offset0);
			cpu.call_stdcall((void*)i64_mod,"rrr",&pcx,&rarg,&pax);
			break;
		case asBC_BAND64:
#ifdef JIT_64
			pax = as<uint64_t>(*edi-offset1);
			pax &= as<uint64_t>(*edi-offset2);
			as<uint64_t>(*edi-offset0) = pax;
#else
			ecx = *edi-offset1;
			edx = *edi-offset1+4;
			ecx &= *edi-offset2;
			edx &= *edi-offset2+4;
			*edi-offset0 = ecx;
			*edi-offset0+4 = edx;
#endif
			break;
		case asBC_BOR64:
#ifdef JIT_64
			pax = as<uint64_t>(*edi-offset1);
			pax |= as<uint64_t>(*edi-offset2);
			as<uint64_t>(*edi-offset0) = pax;
#else
			ecx = *edi-offset1;
			edx = *edi-offset1+4;
			ecx |= *edi-offset2;
			edx |= *edi-offset2+4;
			*edi-offset0 = ecx;
			*edi-offset0+4 = edx;
#endif
			break;
		case asBC_BXOR64:
#ifdef JIT_64
			pax = as<uint64_t>(*edi-offset1);
			pax ^= as<uint64_t>(*edi-offset2);
			as<uint64_t>(*edi-offset0) = pax;
#else
			ecx = *edi-offset1;
			edx = *edi-offset1+4;
			ecx ^= *edi-offset2;
			edx ^= *edi-offset2+4;
			*edi-offset0 = ecx;
			*edi-offset0+4 = edx;
#endif
			break;
		case asBC_BSLL64: {
#ifdef JIT_64
			Register c(cpu, ECX, sizeof(uint64_t) * 8);
			pax = as<uint64_t>(*edi-offset1);
			c = as<uint32_t>(*edi-offset2);
			pax <<= c;
			as<uint64_t>(*edi-offset0) = pax;
#else
			ecx.copy_address(*edi-offset1);
			edx.copy_address(*edi-offset2);
			eax.copy_address(*edi-offset0);
			cpu.call_stdcall((void*)i64_sll,"rrr",&ecx,&edx,&eax);
#endif
			} break;
		case asBC_BSRL64: {
#ifdef JIT_64
			Register c(cpu, ECX, sizeof(uint64_t) * 8);
			pax = as<uint64_t>(*edi-offset1);
			c = as<uint32_t>(*edi-offset2);
			pax.rightshift_logical(c);
			as<uint64_t>(*edi-offset0) = pax;
#else
			ecx.copy_address(*edi-offset1);
			edx.copy_address(*edi-offset2);
			eax.copy_address(*edi-offset0);
			cpu.call_stdcall((void*)i64_srl,"rrr",&ecx,&edx,&eax);
#endif
			} break;
		case asBC_BSRA64: {
#ifdef JIT_64
			Register c(cpu, ECX, sizeof(uint64_t) * 8);
			pax = as<uint64_t>(*edi-offset1);
			c = as<uint32_t>(*edi-offset2);
			pax >>= c;
			as<uint64_t>(*edi-offset0) = pax;
#else
			ecx.copy_address(*edi-offset1);
			edx.copy_address(*edi-offset2);
			eax.copy_address(*edi-offset0);
			cpu.call_stdcall((void*)i64_sra,"rrr",&ecx,&edx,&eax);
#endif
			} break;
		case asBC_CMPi64:
			pcx.copy_address(*edi-offset0);
			pax.copy_address(*edi-offset1);
			cpu.call_stdcall((void*)cmp_int64,"rr",&pcx,&pax);
			ebx = eax;
			break;
		case asBC_CMPu64:
			pcx.copy_address(*edi-offset0);
			pax.copy_address(*edi-offset1);
			cpu.call_stdcall((void*)cmp_uint64,"rr",&pcx,&pax);
			ebx = eax;
			break;
		case asBC_ChkNullS:
			{
			eax = *esi+(asBC_WORDARG0(pOp) * sizeof(asDWORD));
			eax &= eax;
			void* not_zero = cpu.prep_short_jump(NotZero);
			Return(false);
			cpu.end_short_jump(not_zero);
			} break;
		case asBC_ClrHi:
			//Due to the way logic is handled, the upper bytes area always ignored, and don't need to be cleared
			//ebx &= 0x000000ff;
			break;
		case asBC_CallPtr:
			{
				pcx = as<void*>(*pdi-offset0);
				pcx &= pcx;
				auto p = cpu.prep_short_jump(NotZero);
				Return(false);
				cpu.end_short_jump(p);

				pdx = as<void*>(*esp+cpu.stackDepth);
				*pdx + offsetof(asSVMRegisters,programPointer) = pOp+1;
				*pdx + offsetof(asSVMRegisters,stackPointer) = esi;

				pax = as<void*>(*pdx + offsetof(asSVMRegisters,ctx));
				cpu.call_stdcall((void*)callScriptFunction,"rr",&pax,&pcx);
				ReturnFromScriptCall();
			} break;
		//case asBC_FuncPtr: //All pushes are handled above, near asBC_PshC4
		case asBC_LoadThisR:
			{
			pbx = as<void*>(*edi);
			short off = asBC_SWORDARG0(pOp);
			if(off > 0)
				pbx += off;
			else
				pbx -= -off;
			} break;
		//case asBC_PshV8: //All pushes are handled above, near asBC_PshC4
		case asBC_DIVu:
			ecx = *edi-offset2;

			eax = ecx;
			eax &= eax;
			{
			void* zero_test = cpu.prep_short_jump(NotZero);
			Return(false);
			cpu.end_short_jump(zero_test);
			}

			eax = *edi-offset1;
			edx ^= edx;
			ecx.divide();

			*edi-offset0 = eax;
			break;
		case asBC_MODu:
			ecx = *edi-offset2;

			eax = ecx;
			eax &= eax;
			{
			void* zero_test = cpu.prep_short_jump(NotZero);
			Return(false);
			cpu.end_short_jump(zero_test);
			}

			eax = *edi-offset1;
			edx ^= edx;
			ecx.divide();

			*edi-offset0 = edx;
			break;
		case asBC_DIVu64:
			{
#ifdef JIT_64
				pcx = as<uint64_t>(*edi-offset2);

				pax = pcx;
				pax &= pax;
				{
				void* zero_test = cpu.prep_short_jump(NotZero);
				Return(false);
				cpu.end_short_jump(zero_test);
				}

				pax = as<uint64_t>(*edi-offset1);
				pdx ^= pdx;
				pcx.divide();

				as<uint64_t>(*edi-offset0) = pax;
#else
				ecx.copy_address(*edi-offset1);
				edx.copy_address(*edi-offset2);
				eax.copy_address(*edi-offset0);
				cpu.call_stdcall((void*)div_ull,"rrr",&ecx,&edx,&eax);
				eax &= eax;
				auto p = cpu.prep_short_jump(Zero);
				//If 1 is returned, this is a divide by 0 error
				Return(false);
				cpu.end_short_jump(p);
#endif
			} break;
		case asBC_MODu64:
			{
#ifdef JIT_64
				pcx = as<uint64_t>(*edi-offset2);

				pax = pcx;
				pax &= pax;
				{
				void* zero_test = cpu.prep_short_jump(NotZero);
				Return(false);
				cpu.end_short_jump(zero_test);
				}

				pax = as<uint64_t>(*edi-offset1);
				pdx ^= pdx;
				pcx.divide();

				as<uint64_t>(*edi-offset0) = pdx;
#else
				ecx.copy_address(*edi-offset1);
				edx.copy_address(*edi-offset2);
				eax.copy_address(*edi-offset0);
				cpu.call_stdcall((void*)mod_ull,"rrr",&ecx,&edx,&eax);
				eax &= eax;
				auto p = cpu.prep_short_jump(Zero);
				//If 1 is returned, this is a divide by 0 error
				Return(false);
				cpu.end_short_jump(p);
#endif
			} break;
		case asBC_LoadRObjR:
			pbx = as<void*>(*edi-offset0);
			pbx += asBC_SWORDARG1(pOp);
			break;
		case asBC_LoadVObjR:
			pbx.copy_address(*edi+(asBC_SWORDARG1(pOp) - offset0));
			break;
		default:
			//printf("Unhandled op: %i\n", op);
			Return(true);
			break;
		}

		pOp += toSize(op);
	}

	if(waitingForEntry == false)
		Return(true);

	if(tableInUse)
		jumpTables[*output] = (unsigned char**)jumpTable;
	else
		delete[] jumpTable;

	activePage->markUsedAddress((void*)cpu.op);
	jitLock.leave();
	return 0;
}

void asCJITCompiler::finalizePages() {
	jitLock.enter();
	for(auto page = pages.begin(); page != pages.end(); ++page)
		if(!page->second->final)
			page->second->finalize();
	jitLock.leave();
}

void asCJITCompiler::ReleaseJITFunction(asJITFunction func) {
	jitLock.enter();
	auto start = pages.lower_bound(func);

	while(start != pages.end() && start->first == func) {
		start->second->drop();
		start = pages.erase(start);
	}

	auto table = jumpTables.find(func);

	if(table != jumpTables.end()) {
		delete[] table->second;
		jumpTables.erase(table);
	}
	jitLock.leave();
}

unsigned findTotalPushBatchSize(asDWORD* nextOp, asDWORD* endOfBytecode) {
	unsigned bytes = 0;
	while(nextOp < endOfBytecode) {
		asEBCInstr op = (asEBCInstr)*(asBYTE*)nextOp;
		switch(op) {
			case asBC_PUSH:
				bytes += asBC_WORDARG0(nextOp) * sizeof(asDWORD); break;
			case asBC_PshC4:
			case asBC_PshV4:
			case asBC_PshG4:
			case asBC_TYPEID:
				bytes += sizeof(asDWORD); break;
			case asBC_PshV8:
			case asBC_PshC8:
				bytes += sizeof(asQWORD); break;
			case asBC_PSF:
			case asBC_PshVPtr:
			case asBC_PshRPtr:
			case asBC_PshNull:
			case asBC_FuncPtr:
			case asBC_OBJTYPE:
			case asBC_PGA:
			case asBC_VAR:
				bytes += sizeof(void*); break;
			default:
				return bytes;
		}
		nextOp += toSize(op);
	}
	return bytes;
}
void stdcall popStackIfNotEmpty(asIScriptContext* ctx) {
	asCContext* context = (asCContext*)ctx;
	if( context->callStack.GetLength() == 0) {
		context->status = asEXECUTION_FINISHED;
		return;
	}

	context->PopCallState();
}

void stdcall allocScriptObject(asCObjectType* type, asCScriptFunction* constructor, asIScriptEngine* engine, asSVMRegisters* registers) {
	//Allocate and prepare memory
	void* mem = ((asCScriptEngine*)engine)->CallAlloc(type);
	ScriptObject_Construct(type, (asCScriptObject*)mem);

	//Store at address on the stack
	void** dest = *(void***)(registers->stackPointer + constructor->GetSpaceNeededForArguments());
	if(dest)
		*dest = mem;

	//Push pointer so the constructor can be called
	registers->stackPointer -= AS_PTR_SIZE;
	*(void**)registers->stackPointer = mem;

	((asCContext*)registers->ctx)->CallScriptFunction(constructor);
}

void* stdcall engineAlloc(asCScriptEngine* engine, asCObjectType* type) {
	return engine->CallAlloc(type);
}

void stdcall engineRelease(asCScriptEngine* engine, void* memory, asCScriptFunction* release) {
	engine->CallObjectMethod(memory, release->sysFuncIntf, release);
}

void stdcall engineDestroyFree(asCScriptEngine* engine, void* memory, asCScriptFunction* destruct) {
	engine->CallObjectMethod(memory, destruct->sysFuncIntf, destruct);
	engine->CallFree(memory);
}

void stdcall engineFree(asCScriptEngine* engine, void* memory) {
	engine->CallFree(memory);
}

void stdcall engineCallMethod(asCScriptEngine* engine, void* object, asCScriptFunction* method) {
	engine->CallObjectMethod(object, method->sysFuncIntf, method);
}

void stdcall callScriptFunction(asIScriptContext* ctx, asCScriptFunction* func) {
	asCContext* context = (asCContext*)ctx;
	context->CallScriptFunction(func);
}

void stdcall callInterfaceMethod(asIScriptContext* ctx, asCScriptFunction* func) {
	asCContext* context = (asCContext*)ctx;
	context->CallInterfaceMethod(func);
}

size_t stdcall callBoundFunction(asIScriptContext* ctx, unsigned short fid) {
	asCContext* context = (asCContext*)ctx;
	asCScriptEngine* engine = (asCScriptEngine*)context->GetEngine();
	int funcID = engine->importedFunctions[fid]->boundFunctionId;
	if(funcID == -1) {
		context->SetInternalException(TXT_UNBOUND_FUNCTION);
		return 1;
	}
	context->CallScriptFunction(engine->GetScriptFunction(funcID));
	return context->status != asEXECUTION_ACTIVE;
}

asCScriptObject* stdcall castObject(asCScriptObject* obj, asCObjectType* to) {
	asCObjectType *from = obj->objType;
	if( from->DerivesFrom(to) || from->Implements(to) ) {
		obj->AddRef();
		return obj;
	}
	else {
		return nullptr;
	}
}

bool stdcall doSuspend(asIScriptContext* ctx) {
	asCContext* Ctx = (asCContext*)ctx;

	if(Ctx->lineCallback)
		Ctx->CallLineCallback();

	if(Ctx->doSuspend) {
		Ctx->regs.programPointer += 1;
		if(Ctx->status == asEXECUTION_ACTIVE)
			Ctx->status = asEXECUTION_SUSPENDED;
		return true;
	}
	else {
		return false;
	}
}

void SystemCall::callSystemFunction(asCScriptFunction* func, Register* objPointer) {
	auto* sys = func->sysFuncIntf;

	if( sys->takesObjByVal || sys->hasAutoHandles || sys->hostReturnInMemory ||
		(func->returnType.IsObject() && !func->returnType.IsReference()) )
	{
		call_viaAS(func, objPointer);
	}
	else {
		switch(sys->callConv) {
#ifdef JIT_64
		case ICC_CDECL:
		case ICC_STDCALL:
			call_64conv(sys, func, 0, OP_None); break;
		case ICC_CDECL_OBJLAST:
			call_64conv(sys, func, objPointer, OP_Last); break;
		case ICC_CDECL_OBJFIRST:
			call_64conv(sys, func, objPointer, OP_First); break;
		case ICC_THISCALL:
#ifdef _MSC_VER
			call_64conv(sys, func, objPointer, OP_This); break;
#else
			call_64conv(sys, func, objPointer, OP_First); break;
#endif
#else
		case ICC_CDECL:
			call_cdecl(sys, func); break;
		case ICC_STDCALL:
			call_stdcall(sys, func); break;
		case ICC_THISCALL:
			call_thiscall(sys, func, objPointer); break;
		case ICC_CDECL_OBJLAST:
			call_cdecl_obj(sys, func, objPointer, true); break;
		case ICC_CDECL_OBJFIRST:
			call_cdecl_obj(sys, func, objPointer, false); break;
#endif
		default:
			//Probably can't reach here, but handle it anyway
			call_viaAS(func, objPointer); break;
		}
	}
}

void SystemCall::call_entry(asSSystemFunctionInterface* func, asCScriptFunction* sFunc) {
#ifdef JIT_64
	Register esi(cpu,R12,sizeof(void*) * 8);
#else
	Register esi(cpu,ESI,sizeof(void*) * 8);
#endif

	Register eax(cpu,EAX), edx(cpu,EDX), esp(cpu,ESP);
	Register pax(cpu,EAX,sizeof(void*)*8);

	if(!(flags & JIT_SYSCALL_FPU_NORESET))
		fpu.init();

	pax = as<void*>(*esp + cpu.stackDepth);
	as<void*>(*pax + offsetof(asSVMRegisters,programPointer)) = pOp;
	as<void*>(*pax + offsetof(asSVMRegisters,stackPointer)) = esi;

	if(!(flags & JIT_SYSCALL_NO_ERRORS)) {
		pax = as<void*>(*pax + offsetof(asSVMRegisters,ctx));
		pax += offsetof(asCContext,callingSystemFunction); //&callingSystemFunction
		as<void*>(*pax) = sFunc;

		cpu.push(pax); cpu.stackDepth += cpu.pushSize();
	}
}

//Undoes things performed in call_entry in the case of an error
void SystemCall::call_error() {
	if(!(flags & JIT_SYSCALL_NO_ERRORS)) {
		Register pax(cpu,EAX,sizeof(void*)*8);
		cpu.pop(pax);
		as<void*>(*pax) = (void*)0;
	}
}

void SystemCall::call_exit(asSSystemFunctionInterface* func) {
	Register eax(cpu,EAX), esp(cpu,ESP), cl(cpu,ECX,8);
	Register pax(cpu,EAX,sizeof(void*)*8);
			
	if(!(flags & JIT_SYSCALL_NO_ERRORS)) {
		cpu.pop(pax); //IsSystem*
		as<void*>(*pax) = (void*)0;

		cpu.stackDepth -= cpu.pushSize();

		//Check if we should suspend
		pax = as<void*>(*esp+cpu.stackDepth);
		pax = as<void*>(*pax+offsetof(asSVMRegisters,ctx));
		eax = as<int>(*pax+offsetof(asCContext,status));
		eax == (int)asEXECUTION_ACTIVE;
		returnHandler(NotEqual);
	}
}

#ifdef JIT_64
void SystemCall::call_64conv(asSSystemFunctionInterface* func,
		asCScriptFunction* sFunc, Register* objPointer, ObjectPosition pos) {

	Register pax(cpu, EAX, sizeof(void*) * 8), esp(cpu, ESP, sizeof(void*) * 8);
	Register esi(cpu, R12, sizeof(void*) * 8), ebx(cpu, R14, sizeof(void*) * 8);
	Register temp(cpu, EBX, sizeof(void*) * 8);

	call_entry(func, sFunc);

	int argCount = sFunc->parameterTypes.GetLength();
	size_t stackBytes = 0;
	size_t argOffset = 0;
	bool stackObject = false;

	int intCount = 0;
	int floatCount = 0;
	int i = 0, a = 0;

	if(pos == OP_First) {
		++intCount;
		++a;

		if(!cpu.isIntArg64Register(0, 0))
			stackBytes += cpu.pushSize();
	}

	for(; i < argCount; ++i, ++a) {
		auto& type = sFunc->parameterTypes[i];

		if(type.GetTokenType() == ttQuestion) {
			if(!cpu.isIntArg64Register(intCount, a))
				stackBytes += cpu.pushSize();
			++intCount; ++a;
			if(!cpu.isIntArg64Register(intCount, a))
				stackBytes += cpu.pushSize();
			++intCount;

			argOffset += sizeof(void*);
			argOffset += sizeof(int);
		}
		else if(type.IsReference() || type.IsObjectHandle()) {
			if(!cpu.isIntArg64Register(intCount, a))
				stackBytes += cpu.pushSize();
			++intCount;
			argOffset += sizeof(void*);
		}
		else if(type.IsFloatType()) {
			if(!cpu.isFloatArg64Register(intCount, a))
				stackBytes += cpu.pushSize();
			++floatCount;
			argOffset += sizeof(float);
		}
		else if(type.IsDoubleType()) {
			if(!cpu.isFloatArg64Register(intCount, a))
				stackBytes += cpu.pushSize();
			++floatCount;
			argOffset += sizeof(double);
		}
		else if(type.IsPrimitive()) {
			if(!cpu.isIntArg64Register(intCount, a))
				stackBytes += cpu.pushSize();
			++intCount;
			argOffset += type.GetSizeOnStackDWords() * sizeof(asDWORD);
		}
		else {
			throw "Unsupported argument type in system call.";
		}
	}

	--i; --a; --intCount; --floatCount;
	cpu.call_cdecl_prep(stackBytes);

	if(pos != OP_None) {
		if(objPointer) {
			*objPointer &= *objPointer;
			returnHandler(Zero);

			if(pos == OP_First) {
				if(cpu.isIntArg64Register(0, 0)) {
					Register reg = as<void*>(cpu.intArg64(0, 0));
					reg = as<void*>(*objPointer);
				}
				else {
					temp = *objPointer;
				}
			}
			else if(pos == OP_Last) {
				if(cpu.isIntArg64Register(intCount+1, a+1)) {
					Register reg = as<void*>(cpu.intArg64(intCount+1, a+1));
					reg = as<void*>(*objPointer);
				}
				else {
					cpu.push(*objPointer);
				}
			}
			else if(pos == OP_This) {
				Register ecx(cpu, ECX, sizeof(void*) * 8);
				ecx = *objPointer;
			}
		}
		else {
			stackObject = true;

			if(pos == OP_This) {
				Register ecx(cpu, ECX, sizeof(void*) * 8);
				ecx = as<void*>(*esi);
				ecx &= ecx;
				returnHandler(Zero);

				ecx += func->baseOffset;
			}
			else if(pos == OP_First) {
				if(cpu.isIntArg64Register(0, 0)) {
					Register reg = as<void*>(cpu.intArg64(0, 0));
					reg = as<void*>(*esi);
					reg &= reg;
					returnHandler(Zero);

					reg += func->baseOffset;
				}
				else {
					temp = as<void*>(*esi);
					temp &= temp;
					returnHandler(Zero);

					temp += func->baseOffset;
				}
			}
			else if(pos == OP_Last) {
				if(cpu.isIntArg64Register(intCount+1, a+1)) {
					Register reg = as<void*>(cpu.intArg64(intCount+1, a+1));
					reg = as<void*>(*esi);
					reg &= reg;
					returnHandler(Zero);

					reg += func->baseOffset;
				}
				else {
					temp = as<void*>(*esi);
					temp &= temp;
					returnHandler(Zero);

					temp += func->baseOffset;
					cpu.push(temp);
				}
			}

			argOffset += sizeof(void*);
		}

	}

	auto Arg = [&](Register* reg, bool dword) {
		if(dword)
			argOffset -= sizeof(asDWORD);
		else
			argOffset -= sizeof(asQWORD);

		if(reg) {
			if(dword)
				as<asDWORD>(*reg) = as<asDWORD>(*esi+argOffset);
			else
				as<asQWORD>(*reg) = as<asQWORD>(*esi+argOffset);
		}
		else {
			if(dword)
				cpu.push(as<asDWORD>(*esi+argOffset));
			else
				cpu.push(as<asDWORD>(*esi+argOffset));
		}
	};

	auto IntArg = [&](bool dword) {
		if(cpu.isIntArg64Register(intCount, a)) {
			Register arg = cpu.intArg64(intCount, a);
			Arg(&arg, dword);
		}
		else
			Arg(0, dword);
		--intCount;
	};

	auto FloatArg = [&](bool dword) {
		if(cpu.isFloatArg64Register(floatCount, a)) {
			Register arg = cpu.floatArg64(floatCount, a);
			Arg(&arg, dword);
		}
		else
			Arg(0, dword);
		--floatCount;
	};


	for(; i >= 0; --i, --a) {
		auto& type = sFunc->parameterTypes[i];

		if(type.GetTokenType() == ttQuestion) {
			IntArg(true);
			IntArg(false);
		}
		else if(type.IsReference() || type.IsObjectHandle()) {
			IntArg(false);
		}
		else if(type.IsFloatType()) {
			FloatArg(true);
		}
		else if(type.IsDoubleType()) {
			FloatArg(false);
		}
		else if(type.IsPrimitive()) {
			IntArg(type.GetSizeOnStackDWords() == 1);
		}
	}

	if(pos == OP_First && !cpu.isIntArg64Register(0, 0))
		cpu.push(temp);

	cpu.call((void*)func->func);

	cpu.call_cdecl_end(stackBytes);

	if(stackObject)
		esi += func->paramSize * sizeof(asDWORD) + sizeof(void*);
	else if(func->paramSize > 0)
		esi += func->paramSize * sizeof(asDWORD);

	if(func->hostReturnSize > 0) {
		if(func->hostReturnFloat) {
			Register ret = cpu.floatReturn64();
			if(func->hostReturnSize == 1) {
				esp -= cpu.pushSize();
				as<float>(*esp) = as<float>(ret);
				as<float>(ebx) = as<float>(*esp);
				esp += cpu.pushSize();
			}
			else {
				esp -= cpu.pushSize();
				as<double>(*esp) = as<double>(ret);
				cpu.pop(ebx);
			}
		}
		else {
			if(func->hostReturnSize == 1)
				as<uint32_t>(ebx) = as<uint32_t>(cpu.intReturn64());
			else
				as<uint64_t>(ebx) = as<uint64_t>(cpu.intReturn64());
		}
	}

	call_exit(func);
}
#else
void SystemCall::call_getPrimitiveReturn(asSSystemFunctionInterface* func) {
	Register eax(cpu,EAX), ebx(cpu,EBX), edx(cpu,EDX), ebp(cpu,EBP), esp(cpu,ESP);
	if(func->hostReturnSize > 0) {
		if(func->hostReturnFloat) {
			if(func->hostReturnSize == 1) {
				esp -= cpu.pushSize();
				fpu.store_float(*esp);
				cpu.pop(ebx);
			}
			else {
				esp -= sizeof(double);
				fpu.store_double(*esp);
				cpu.pop(ebx); cpu.pop(ebp);
			}
		}
		else {
			if(func->hostReturnSize == 1) {
				ebx = eax;
			}
			else {
				ebx = eax; ebp = edx;
			}
		}
	}
}

void SystemCall::call_stdcall(asSSystemFunctionInterface* func, asCScriptFunction* sFunc) {
	Register eax(cpu,EAX), ebx(cpu,EBX), ebp(cpu,EBP), edx(cpu,EDX), esp(cpu,ESP), esi(cpu,ESI);
	Register cl(cpu,ECX,8);

	call_entry(func,sFunc);

	int firstArg = 0, lastArg = func->paramSize,
		argBytes = (lastArg-firstArg) * cpu.pushSize();

	for(int i = lastArg-1; i >= firstArg; --i)
		cpu.push(*esi+(i*sizeof(asDWORD)));

	cpu.call((void*)func->func);

	unsigned popCount = func->paramSize * sizeof(asDWORD);
	if(popCount > 0)
		esi += popCount;

	call_getPrimitiveReturn(func);

	call_exit(func);
}
	
void SystemCall::call_cdecl(asSSystemFunctionInterface* func, asCScriptFunction* sFunc) {
	Register eax(cpu,EAX), ebx(cpu,EBX), ebp(cpu,EBP), edx(cpu,EDX), esp(cpu,ESP), esi(cpu,ESI);
	Register cl(cpu,ECX,8);

	call_entry(func,sFunc);

	int firstArg = 0, lastArg = func->paramSize, argBytes;

	argBytes = (lastArg-firstArg) * cpu.pushSize();
	cpu.call_cdecl_prep(argBytes);

	for(int i = lastArg-1; i >= firstArg; --i)
		cpu.push(*esi+(i*sizeof(asDWORD)));

	cpu.call((void*)func->func);
	cpu.call_cdecl_end(argBytes);

	unsigned popCount = func->paramSize * sizeof(asDWORD);
	if(popCount > 0)
		esi += popCount;

	call_getPrimitiveReturn(func);

	call_exit(func);
}


void SystemCall::call_cdecl_obj(asSSystemFunctionInterface* func, asCScriptFunction* sFunc, Register* objPointer, bool last) {
	Register eax(cpu,EAX), ebx(cpu,EBX), ecx(cpu,ECX), ebp(cpu,EBP), edx(cpu,EDX), esp(cpu,ESP), esi(cpu,ESI);
	Register cl(cpu,ECX,8);

	call_entry(func,sFunc);

	int firstArg = 0, lastArg = func->paramSize, argBytes;

	argBytes = (lastArg-firstArg + 1) * cpu.pushSize();
	cpu.call_cdecl_prep(argBytes);

	if(objPointer) {
		*objPointer &= *objPointer;

		auto j = cpu.prep_short_jump(NotZero);
		call_error();
		returnHandler(Jump);
		cpu.end_short_jump(j);

		if(last)
			cpu.push(*objPointer);
	}
	else {
		firstArg = 1; lastArg += 1;
		ecx = as<void*>(*esi);
		ecx &= ecx;
		returnHandler(Zero);
		ecx += func->baseOffset;
		if(last)
			cpu.push(ecx);
	}

	for(int i = lastArg-1; i >= firstArg; --i)
		cpu.push(*esi+(i*sizeof(asDWORD)));

	if(!last)
		if(objPointer)
			cpu.push(*objPointer);
		else
			cpu.push(ecx);

	cpu.call((void*)func->func);
	cpu.call_cdecl_end(argBytes);

	unsigned popCount = func->paramSize * sizeof(asDWORD);
	if(!objPointer)
		popCount += sizeof(void*);
	if(popCount > 0)
		esi += popCount;

	call_getPrimitiveReturn(func);

	call_exit(func);
}

void SystemCall::call_thiscall(asSSystemFunctionInterface* func, asCScriptFunction* sFunc, Register* objPointer) {
	Register eax(cpu,EAX), ebx(cpu,EBX), ecx(cpu,ECX), ebp(cpu,EBP), edx(cpu,EDX), esp(cpu,ESP), esi(cpu,ESI);
	Register cl(cpu,ECX,8);

	call_entry(func,sFunc);

	int firstArg = 0, lastArg = func->paramSize, argBytes;

	if(objPointer) {
		*objPointer &= *objPointer;
		auto j = cpu.prep_short_jump(NotZero);
		call_error();
		returnHandler(Jump);
		cpu.end_short_jump(j);
	}
	else {
		ecx = as<void*>(*esi);
		firstArg = 1; lastArg += 1;

		ecx &= ecx;
		auto j = cpu.prep_short_jump(NotZero);

		call_error();
		returnHandler(Jump);

		cpu.end_short_jump(j);

		ecx += func->baseOffset;
	}
	argBytes = (lastArg-firstArg) * cpu.pushSize();
	cpu.call_thiscall_prep(argBytes);

	for(int i = lastArg-1; i >= firstArg; --i)
		cpu.push(*esi+(i*sizeof(asDWORD)));

	if(objPointer)
		cpu.call_thiscall_this(*objPointer);
	else
		cpu.call_thiscall_this(*esi);
	cpu.call((void*)func->func);
	cpu.call_thiscall_end(argBytes);

	unsigned popCount = func->paramSize * sizeof(asDWORD);
	if(!objPointer)
		popCount += sizeof(void*);
	if(popCount > 0)
		esi += popCount;

	call_getPrimitiveReturn(func);

	call_exit(func);
}
#endif

void SystemCall::call_viaAS(asCScriptFunction* func, Register* objPointer) {
#ifdef JIT_64
	Register esi(cpu,R12,sizeof(void*) * 8);
#else
	Register esi(cpu,ESI,sizeof(void*) * 8);
#endif

	Register eax(cpu,EAX), ebx(cpu,EBX), ecx(cpu,ECX), edx(cpu,EDX), esp(cpu,ESP), ebp(cpu,EBP), edi(cpu,EDI);
	Register pax(cpu,EAX,sizeof(void*) * 8), pcx(cpu,ECX,sizeof(void*) * 8), pdx(cpu,EDX,sizeof(void*) * 8),
		pbx(cpu,EBX,sizeof(void*) * 8);
	Register cl(cpu,ECX,8);

	//Copy state to VM state in case the call inspects the context
	call_entry(func->sysFuncIntf,func);

	if(!(flags & JIT_SYSCALL_NO_ERRORS))
		pax = as<void*>(*esp + cpu.stackDepth);
	MemAddress ctxPtr(as<void*>(*pax + offsetof(asSVMRegisters,ctx)));

	int stdcall callSysWrapper(int id, asIScriptContext* ctx, void* obj);

	if(objPointer)
		cpu.call_stdcall((void*)callSysWrapper,"cmr",func->GetId(),&ctxPtr,objPointer);
	else
		cpu.call_stdcall((void*)callSysWrapper,"cmc",func->GetId(),&ctxPtr,nullptr);
	esi += pax;

	MemAddress regptr( as<void*>( *esp + cpu.stackDepth ) );
		
	int stdcall sysExit(asSVMRegisters* registers);
	cpu.call_stdcall((void*)sysExit,"m",&regptr);

	//Check that there is a return in the valueRegister
	bool isGeneric = func->sysFuncIntf->callConv == ICC_GENERIC_FUNC || func->sysFuncIntf->callConv == ICC_GENERIC_FUNC_RETURNINMEM
		|| func->sysFuncIntf->callConv == ICC_GENERIC_METHOD || func->sysFuncIntf->callConv == ICC_GENERIC_METHOD_RETURNINMEM;

	if(((func->sysFuncIntf->hostReturnSize >= 1 && !func->sysFuncIntf->hostReturnInMemory) || isGeneric)
	 	&& !(func->returnType.IsObject() && !func->returnType.IsReference()) ) {

		pcx = as<void*>(*esp+cpu.stackDepth);
#ifdef JIT_64
		Register valueReg(cpu, R14, sizeof(void*) * 8);
		valueReg = as<asQWORD>(*pcx + offsetof(asSVMRegisters,valueRegister));
#else
		ebx = *pcx + offsetof(asSVMRegisters,valueRegister);

		if(func->sysFuncIntf->hostReturnSize >= 2 || isGeneric)
			ebp = *ecx + offsetof(asSVMRegisters,valueRegister)+4;
#endif
	}

	call_exit(func->sysFuncIntf);
}

int stdcall sysExit(asSVMRegisters* registers) {
	if(registers->doProcessSuspend) {
		asCContext* context = (asCContext*)registers->ctx;
		if(context->status != asEXECUTION_ACTIVE) {
			return 1;
		}
		else if(context->doSuspend) {
			context->status = asEXECUTION_SUSPENDED;
			return 1;
		}
	}
	return 0;
}

int stdcall callSysWrapper(int id, asIScriptContext* ctx, void* obj) {
	return CallSystemFunction(id, (asCContext*)ctx, obj) * sizeof(asDWORD);
}
