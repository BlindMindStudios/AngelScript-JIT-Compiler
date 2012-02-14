#ifdef WIN_MODE
#include "virtual_asm.h"
#include <Windows.h>

namespace assembler {

CodePage::CodePage(unsigned int Size, void* requestedStart) : used(0), final(false), references(1) {
	unsigned minPageSize = getMinimumPageSize();
	unsigned pages = Size / minPageSize;
	if(Size % minPageSize != 0)
		pages += 1;

	size = pages * minPageSize;

	page = VirtualAlloc(requestedStart, size, MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE);
}

void CodePage::grab() {
	++references;
}

void CodePage::drop() {
	if(--references == 0)
		delete this;
}

CodePage::~CodePage() {
	VirtualFree(page,0,MEM_RELEASE);
}

void CodePage::finalize() {
	FlushInstructionCache(GetCurrentProcess(),page,size);
	DWORD oldProtect = PAGE_EXECUTE_READWRITE;
	VirtualProtect(page,size,PAGE_EXECUTE_READ,&oldProtect);
	final = true;
}

unsigned int CodePage::getMinimumPageSize() {
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return info.dwPageSize;
}


void CriticalSection::enter() {
	EnterCriticalSection((CRITICAL_SECTION*)pLock);
}

void CriticalSection::leave() {
	LeaveCriticalSection((CRITICAL_SECTION*)pLock);
}

CriticalSection::CriticalSection() {
	auto* section = new CRITICAL_SECTION;
	InitializeCriticalSection(section);
	pLock = section;
}
CriticalSection::~CriticalSection() {
	DeleteCriticalSection((CRITICAL_SECTION*)pLock);
}

};
#endif
